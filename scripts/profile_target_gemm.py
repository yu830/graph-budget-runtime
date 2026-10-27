"""One located GEMM launch for a bounded Nsight Compute collection."""
import fcntl
import os
import torch
from graphbudget.experiment import gpu_lease
from sglang.srt.batch_invariant_ops.batch_invariant_ops import _matmul_persistent_triton

with open(os.environ['GRAPHBUDGET_SHARED_GPU_LOCK'],'a') as lock,gpu_lease():
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    a=torch.full((4,4864),.0625,device='cuda',dtype=torch.bfloat16)
    b=torch.full((896,4864),.125,device='cuda',dtype=torch.bfloat16).T
    output=_matmul_persistent_triton(a,b)
    torch.cuda.synchronize()
    assert torch.equal(output,torch.full_like(output,38.))
    print('PASS exact simple oracle; geometry',os.environ.get('GRAPHBUDGET_SMALL_M_GEMM','0'))
