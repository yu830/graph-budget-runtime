"""Fresh-server matched workloads. No profiler or diagnostic events in timed runs."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import uuid

from graphbudget.benchmark import run_benchmark, compare_exact_tokens, load_workload
from graphbudget.experiment import gpu_lease
from graphbudget.lifecycle import http_server, OwnedProcess, stop_owned
from graphbudget.runtime import write_json
from profile_metadata import server_args, post, workload

ROOT=Path(__file__).resolve().parents[1]
MODES=('original','native','prealloc','fused','prefix_only','gemm','upstream_gemm','v2_residual')

def make_workloads(out):
    source=json.loads((ROOT/'examples/workloads/calibration.json').read_text())
    rows=source['requests']
    primary={**source,'workload_id':'metadata-primary-256x128-v1','purpose':'formal_primary',
             'requests':[{**rows[i%len(rows)],'request_id':f'primary-{i:03}',
                          'planned_at_s':0,'max_new_tokens':128} for i in range(256)]}
    held=[]
    for wave,n in enumerate((3,5,9,3,9,5)):
        for i in range(n):
            row=rows[(wave*9+i)%len(rows)]
            held.append({**row,'request_id':f'holdout-{wave}-{i}',
                         'input_ids':row['input_ids']*(2 if wave%2 else 3),
                         'planned_at_s':wave*3,'max_new_tokens':64 if i%2 else 96})
    holdout={**source,'workload_id':'metadata-holdout-bursts-v1','purpose':'heldout', 'requests':held}
    write_json(out/'primary.json',primary);write_json(out/'holdout.json',holdout)
    return primary,holdout

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--mode',choices=MODES,required=True)
    p.add_argument('--rounds',type=int,default=1);p.add_argument('--screen',action='store_true')
    p.add_argument('--reference');p.add_argument('--reference-dir')
    p.add_argument('--model-path');p.add_argument('--max-running-requests',type=int,default=16)
    p.add_argument('--max-total-tokens',type=int,default=4096);p.add_argument('--max-workers',type=int,default=32)
    p.add_argument('--port',type=int,default=30277)
    p.add_argument('--extra-workload',action='append',nargs=3,default=[],metavar=('LABEL','PATH','WORKERS'))
    a=p.parse_args()
    if a.max_workers < 1: raise ValueError('max-workers must be positive')
    if a.reference and a.reference_dir: raise ValueError('use either reference or reference-dir')
    out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=False)
    primary,holdout=make_workloads(out)
    if a.screen:
        primary['requests']=primary['requests'][:64]
        primary['workload_id']='metadata-screen-64x128-v1'
        primary['purpose']='calibration'
        write_json(out/'primary.json',primary)
    workloads=[('primary',primary,a.max_workers),('holdout',holdout,a.max_workers)]
    used={'primary','holdout'}
    for label,path,workers_text in a.extra_workload:
        if not re.fullmatch(r'[a-z0-9][a-z0-9_-]*',label) or label in used:
            raise ValueError(f'invalid or duplicate workload label: {label}')
        workers=int(workers_text)
        if workers < 1: raise ValueError('workload workers must be positive')
        workload_data=load_workload(path).to_dict()
        write_json(out/f'{label}-workload.json',workload_data)
        workloads.append((label,workload_data,workers));used.add(label)
    env=dict(os.environ)
    env.pop('GRAPHBUDGET_METADATA_DIAGNOSTIC',None)
    env['GRAPHBUDGET_METADATA_MODE']=a.mode if a.mode in ('prealloc','fused','prefix_only') else 'native'
    env['GRAPHBUDGET_SMALL_M_GEMM']={'gemm':'1','upstream_gemm':'upstream_42145','v2_residual':'upstream_42145'}.get(a.mode,'0')
    env['GRAPHBUDGET_RESIDUAL_NORM']='1' if a.mode == 'v2_residual' else '0'
    if a.mode=='original':
        paths=env.get('PYTHONPATH','').split(os.pathsep)
        replacements=(
            (ROOT/'src', Path(env['GRAPHBUDGET_BASELINE_ROOT'])/'src'),
            (Path(env['SGLANG_SOURCE'])/'python', Path(env['GRAPHBUDGET_BASELINE_SGLANG_SOURCE'])/'python'),
        )
        for current, original in replacements:
            current, original = str(current.resolve()), str(original.resolve())
            if paths.count(current) != 1 or not Path(original).is_dir():
                raise ValueError(f'Cannot bind original source path: {current} -> {original}')
            paths[paths.index(current)] = original
        env['PYTHONPATH']=os.pathsep.join(paths)
        env.pop('GRAPHBUDGET_METADATA_MODE',None);env.pop('GRAPHBUDGET_SMALL_M_GEMM',None);env.pop('GRAPHBUDGET_RESIDUAL_NORM',None)
    run=uuid.uuid4().hex
    cmd=[sys.executable,'-m','graphbudget.cli','_serve','--output',str(out/'runtime'),
         '--run-id',run,'--policy','native','--observe','counts','--',
         *server_args(a.model_path,a.max_running_requests,a.max_total_tokens,a.port)]
    write_json(out/'invocation.json',{'command':cmd,'mode':a.mode,'run_id':run,
               'pythonpath':env.get('PYTHONPATH',''),
               'workloads':[{'label':label,'workers':workers,'requests':len(w['requests'])} for label,w,workers in workloads],
               'settings':{k:env.get(k) for k in ('GRAPHBUDGET_METADATA_MODE','GRAPHBUDGET_SMALL_M_GEMM','GRAPHBUDGET_RESIDUAL_NORM','GRAPHBUDGET_METADATA_DIAGNOSTIC')}})
    results=[];telemetry=None
    with open(os.environ['GRAPHBUDGET_SHARED_GPU_LOCK'],'a') as lock,gpu_lease(),(out/'server.log').open('w') as log,(out/'device.csv').open('w') as device:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            telemetry=OwnedProcess(['nvidia-smi','--query-gpu=timestamp,memory.used,temperature.gpu,power.draw,clocks.sm,clocks.mem,utilization.gpu','--format=csv,noheader,nounits','-lms','250'],stdout=device,stderr=subprocess.STDOUT)
            with http_server(cmd,out/'runtime',run,timeout=600,stdout=log,env=env) as (_,life):
                url=life['url'];ready=json.loads((out/'runtime/ready.json').read_text())
                expected_capture=sorted({n for n in (1,2,4,8,12,16) if n<=a.max_running_requests}|{a.max_running_requests})
                assert ready['kv_tokens']==a.max_total_tokens and ready['capture_sizes']==expected_capture
                assert ready['identity']['config']['max_running_requests']==a.max_running_requests
                assert ready['identity']['attention_backend']=='triton'
                warmed=run_benchmark(url,workload(4,64),output=out/'warmup.json',run_id=run+'-warm',reset_cache=True)
                assert warmed['summary']['completed_requests']==4
                reference=json.loads((ROOT/'examples/workloads/calibration.json').read_text())
                check=run_benchmark(url,reference,output=out/'correctness.json',run_id=run+'-correctness',reset_cache=True)
                assert check['summary']['completed_requests']==43
                reference_path=Path(a.reference_dir)/'correctness.json' if a.reference_dir else a.reference
                if reference_path:
                    comparison=compare_exact_tokens(json.loads(Path(reference_path).read_text()),check)
                    write_json(out/'exact-reference.json',comparison)
                    assert comparison['equivalent'],comparison
                for r in range(a.rounds):
                    for label,w,workers in workloads:
                        post(url+'/flush_cache')
                        paths=json.loads((out/'runtime/paths.json').read_text())
                        write_json(out/f'{label}-{r}-paths-before.json',paths)
                        bench=run_benchmark(url,w,output=out/f'{label}-{r}.json',run_id=f'{run}-{label}-{r}',reset_cache=True,max_workers=workers,timeout_s=240)
                        assert bench['summary']['completed_requests']==len(w['requests']),bench['summary']
                        if a.reference_dir:
                            reference_run=json.loads((Path(a.reference_dir)/f'{label}-{r}.json').read_text())
                            comparison=compare_exact_tokens(reference_run,bench)
                            write_json(out/f'{label}-{r}-exact-reference.json',comparison)
                            assert comparison['equivalent'],comparison
                        post(url+'/flush_cache')
                        paths=json.loads((out/'runtime/paths.json').read_text())
                        write_json(out/f'{label}-{r}-paths-after.json',paths)
                        results.append({'label':label,'round':r,'summary':bench['summary']})
                        print(a.mode,label,r,json.dumps(bench['summary']),flush=True)
                write_json(out/'result.json',{'status':'PASS','mode':a.mode,'identity':ready['identity'],
                    'fingerprint':ready['fingerprint'],'lifecycle':life,'results':results,
                    'scope':'one fixed static capture set; measured identity, no stale plans reused'})
        finally: stop_owned(telemetry)

if __name__=='__main__':main()
