"""Reuse identical rows across batch sizes/order and both candidate geometries."""
import fcntl
import json
import os
from pathlib import Path
import sys
import torch
from graphbudget.experiment import gpu_lease
from graphbudget.runtime import write_json
from check_small_m_gemm import call, SHAPES

out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=False)
checks=[]
with open(os.environ['GRAPHBUDGET_SHARED_GPU_LOCK'],'a') as lock,gpu_lease():
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    torch.manual_seed(277)
    for k,n in SHAPES:
        a=torch.randn((17,k),device='cuda',dtype=torch.bfloat16)
        b=torch.randn((n,k),device='cuda',dtype=torch.bfloat16).T
        bias=torch.randn((n,),device='cuda',dtype=torch.bfloat16)
        ref=call(a,b,bias,0)
        for flag in ('0','1','upstream_42145'):
            for m in range(1,18):
                forward=call(a[:m],b,bias,flag)
                ids=torch.arange(m-1,-1,-1,device='cuda')
                reverse=call(a[:m][ids],b,bias,flag)
                assert torch.equal(forward,ref[:m]),(flag,m,k,n,'prefix')
                assert torch.equal(reverse,ref[:m][ids]),(flag,m,k,n,'reorder')
                for i in (0,m-1):
                    single=call(a[i:i+1],b,bias,flag)
                    assert torch.equal(single,ref[i:i+1]),(flag,m,k,n,'single')
                checks.append({'mode':flag,'M':m,'K':k,'N':n,'prefix':True,'reverse':True,'single_first_last':True,'random_bias':True})
        print('invariant',k,n,flush=True)
    write_json(out/'result.json',{'status':'PASS','checks':checks,'bitwise':True})
