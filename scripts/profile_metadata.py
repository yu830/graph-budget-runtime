"""Short native-worker profile using the existing SGLang profiling API."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request
import uuid

from graphbudget.benchmark import run_benchmark
from graphbudget.experiment import gpu_lease
from graphbudget.lifecycle import http_server
from graphbudget.runtime import write_json

ROOT = Path(__file__).resolve().parents[1]
def server_args(model_path=None, max_running_requests=16, max_total_tokens=4096, port=30277):
    if not 1 <= max_running_requests <= 16 or max_total_tokens < 1:
        raise ValueError("max running requests must be 1..16 and token capacity positive")
    model = str(Path(model_path or os.environ['GRAPHBUDGET_MODEL_PATH']).resolve())
    capture_sizes = sorted({n for n in (1, 2, 4, 8, 12, 16) if n <= max_running_requests}
                           | {max_running_requests})
    return ['--model-path', model, '--tokenizer-path', model, '--revision', Path(model).name,
            '--dtype', 'bfloat16', '--device', 'cuda', '--tensor-parallel-size', '1',
            '--attention-backend', 'triton', '--disable-overlap-schedule', '--stream-output',
            '--random-seed', '0', '--enable-deterministic-inference',
            '--weight-loader-disable-mmap',
            '--mem-fraction-static', '.85', '--max-total-tokens', str(max_total_tokens),
            '--max-running-requests', str(max_running_requests), '--context-length', '4096',
            '--cuda-graph-bs', *map(str, capture_sizes), '--host', '127.0.0.1', '--port', str(port)]

def post(url, obj=None):
    data = None if obj is None else json.dumps(obj).encode()
    with urllib.request.urlopen(urllib.request.Request(url, data=data, method='POST',
                                headers={'Content-Type':'application/json'}), timeout=120) as r:
        return r.read().decode()

def workload(n=4, length=64):
    source = json.loads((ROOT/'examples/workloads/calibration.json').read_text())
    source['requests'] = [{**row, 'request_id':f'profile-{i}', 'planned_at_s':0,
                           'max_new_tokens':length} for i,row in enumerate(source['requests'][:n])]
    return source

def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',required=True)
    p.add_argument('--nsys',action='store_true'); p.add_argument('--events-only',action='store_true'); a=p.parse_args()
    out=Path(a.output).resolve(); out.mkdir(parents=True, exist_ok=False)
    run=uuid.uuid4().hex
    cmd=[sys.executable,'-m','graphbudget.cli','_serve','--output',str(out/'runtime'),
         '--run-id',run,'--policy','native','--observe','counts','--',*server_args()]
    write_json(out/'invocation.json', {'command':cmd,'run_id':run,'purpose':'diagnosis only'})
    with open(os.environ['GRAPHBUDGET_SHARED_GPU_LOCK'],'a') as lock, gpu_lease(), (out/'server.log').open('w') as log:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        with http_server(cmd,out/'runtime',run,timeout=600,stdout=log) as (_,life):
            url=life['url']; w=workload()
            warm=run_benchmark(url,w,output=out/'warmup.json',run_id=run+'-warm',reset_cache=True)
            assert warm['summary']['completed_requests']==4
            if not a.events_only:
                receipt=post(url+'/start_profile', {'output_dir':str(out/'trace'),'activities':['CUDA_PROFILER'] if a.nsys else ['CPU','GPU'],
                         'with_stack':False,'record_shapes':True,'profile_prefix':'metadata-native'})
                write_json(out/'start_profile.json',{'response':receipt})
            bench=run_benchmark(url,w,output=out/'requests.json',run_id=run,reset_cache=True)
            if not a.events_only:
                receipt=post(url+'/stop_profile')
                write_json(out/'stop_profile.json',{'response':receipt})
            post(url+'/flush_cache')
            write_json(out/'result.json',{'benchmark':bench['summary'],'lifecycle':life})
            print(json.dumps(bench['summary']),flush=True)

if __name__=='__main__': main()
