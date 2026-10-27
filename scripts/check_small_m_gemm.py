"""Validate one opt-in tile specialization of the existing deterministic GEMM."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import random
import statistics
import time
import torch
from graphbudget.experiment import gpu_lease
from graphbudget.runtime import write_json
from graphbudget.identity import file_digest
import inspect
import sglang.srt.batch_invariant_ops.batch_invariant_ops as ops
_matmul_persistent_triton = ops._matmul_persistent_triton

SHAPES=((896,896),(896,1152),(896,9728),(4864,896),(896,151936))
def call(a,b,bias,flag):
    # Test-only switching; real workers freeze the setting before capture.
    ops._GRAPHBUDGET_SMALL_M_GEMM=str(flag)
    return _matmul_persistent_triton(a,b,bias)

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--candidate',default='1');args=p.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    write_json(out/'source.json',{'script_sha256':file_digest(__file__),
             'sglang_source_file':inspect.getsourcefile(_matmul_persistent_triton),
             'source_sha256':file_digest(inspect.getsourcefile(_matmul_persistent_triton)),
             'candidate':args.candidate})
    checks=[];samples=[]
    with open(os.environ['GRAPHBUDGET_SHARED_GPU_LOCK'],'a') as lock,gpu_lease():
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        torch.manual_seed(727)
        for m in range(1,18):
            for k,n in SHAPES:
                for distribution in ('small_exact','normal'):
                    if distribution=='small_exact':
                        a_cpu=torch.randint(-2,3,(m,k),dtype=torch.int32)
                        b_cpu=torch.randint(-2,3,(n,k),dtype=torch.int32).T
                        a=a_cpu.to('cuda',torch.bfloat16)/16;b=b_cpu.to('cuda',torch.bfloat16)/16
                        # CPU float64 is exact for these bounded integer/256 sums.
                        oracle=(a_cpu.double()@b_cpu.double()/256).to(torch.bfloat16)
                    else:
                        a=torch.randn((m,k),device='cuda',dtype=torch.bfloat16)
                        b=torch.randn((n,k),device='cuda',dtype=torch.bfloat16).T
                        oracle=None
                    for bias_on in (False,True):
                        bias=torch.full((n,),.25,device='cuda',dtype=torch.bfloat16) if bias_on else None
                        native=call(a,b,bias,0);cand=call(a,b,bias,args.candidate);torch.cuda.synchronize()
                        if not torch.equal(native,cand):
                            i,j=(native!=cand).nonzero()[0].tolist()
                            write_json(out/'failure.json', {'M':m,'K':k,'N':n,'distribution':distribution,'bias':bias_on,
                                       'candidate':args.candidate,'first_index':[i,j],'native':native[i,j].item(),'candidate_value':cand[i,j].item()})
                            raise AssertionError(f'native mismatch {m,k,n,distribution,bias_on}')
                        if oracle is not None:
                            exact=oracle if not bias_on else ((a_cpu.double()@b_cpu.double()/256)+.25).to(torch.bfloat16)
                            assert torch.equal(cand.cpu(),exact),f'oracle mismatch {m,k,n,bias_on}'
                        checks.append({'M':m,'K':k,'N':n,'distribution':distribution,'bias':bias_on,
                                       'bitwise_native':True,'independent_oracle':oracle is not None,
                                       'a_stride':a.stride(),'b_stride':b.stride(),'dtype':'bfloat16'})
                if m in (1,4,16):
                    for flag in (0,args.candidate):
                        for _ in range(10):call(a,b,None,flag)
                    for round_id in range(5):
                        for flag in ((0,args.candidate,args.candidate,0) if round_id%2==0 else (args.candidate,0,0,args.candidate)):
                            start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                            torch.cuda.synchronize();start.record()
                            for _ in range(100):call(a,b,None,flag)
                            end.record();end.synchronize()
                            samples.append({'M':m,'K':k,'N':n,'round':round_id,'flag':flag,'us':start.elapsed_time(end)*10})
                write_json(out/'correctness.json',{'checks':checks,'status':'RUNNING'})
                print('checked',m,k,n,flush=True)
        write_json(out/'correctness.json',{'checks':checks,'status':'PASS'})
        write_json(out/'raw.json',{'samples':samples,'scope':'complete upstream GEMM call event spans, no profiler'})
        for m in (1,4,16):
            for k,n in SHAPES:
                print(m,k,n,{f:statistics.median(r['us'] for r in samples if r['M']==m and r['K']==k and r['N']==n and r['flag']==f) for f in (0,args.candidate)},flush=True)
if __name__=='__main__':main()
