#!/usr/bin/env python3
"""Exercise the actual patched RMSNorm dispatch and required fallbacks on GPU."""
import argparse,fcntl,os,sys,types
from pathlib import Path
from unittest.mock import patch
import torch
from graphbudget.experiment import gpu_lease
from graphbudget.identity import file_digest
from graphbudget.runtime import write_json
os.environ['GRAPHBUDGET_RESIDUAL_NORM']='1'
import graphbudget.fused_norm as fused
import sglang.srt.layers.layernorm as layernorm
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',required=True)
out=Path(parser.parse_args().output);out.mkdir(parents=True,exist_ok=False)
result={'status':'RUNNING','graphbudget_source_sha256':file_digest(fused.__file__),'sglang_source_sha256':file_digest(layernorm.__file__),'checks':[]}
def bit_equal(a,b):return torch.equal(a.contiguous().view(torch.int16),b.contiguous().view(torch.int16))
try:
 with open(os.environ['GRAPHBUDGET_SHARED_GPU_LOCK'],'a') as lock,gpu_lease():
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);torch.manual_seed(891);torch.set_grad_enabled(False)
  with patch.object(layernorm,'is_batch_invariant_mode_enabled',return_value=True),patch.object(layernorm,'get_global_server_args',return_value=types.SimpleNamespace(rl_on_policy_target='')):
   layer=layernorm.RMSNorm(896,eps=1e-6).to(device='cuda',dtype=torch.bfloat16)
   layer.weight.data.copy_(torch.randn_like(layer.weight))
   for m in (1,7,8,12,16,17):
    x=torch.randn((m,896),device='cuda',dtype=torch.bfloat16);res=torch.randn_like(x)
    reference=layer.forward_native(x,res)
    with patch.object(layernorm,'_graphbudget_residual_norm',wraps=fused.residual_rms_norm) as called:
     output=layer.forward_cuda(x,res)
     assert called.call_count==(1 if m<=16 else 0),(m,called.call_count)
    assert all(bit_equal(a,b) for a,b in zip(reference,output)),m
    result['checks'].append({'case':'M boundary','M':m,'fast_path':m<=16})
   x=torch.randn((8,1792),device='cuda',dtype=torch.bfloat16)[:,::2];res=torch.randn((8,896),device='cuda',dtype=torch.bfloat16)
   with patch.object(layernorm,'_graphbudget_residual_norm',wraps=fused.residual_rms_norm) as called:
    output=layer.forward_cuda(x,res);assert called.call_count==0
   assert all(bit_equal(a,b) for a,b in zip(layer.forward_native(x,res),output))
   result['checks'].append({'case':'noncontiguous fallback'})
   x=torch.randn((8,896),device='cuda',dtype=torch.bfloat16);res=torch.randn_like(x);extra=torch.randn_like(x)
   with patch.object(layernorm,'_graphbudget_residual_norm',wraps=fused.residual_rms_norm) as called:
    output=layer.forward_cuda(x,res,extra);assert called.call_count==0
   assert all(bit_equal(a,b) for a,b in zip(layer.forward_native(x,res,extra),output))
   result['checks'].append({'case':'post residual fallback'})
   with patch.object(fused,'_verified_device',return_value=False),patch.object(layernorm,'_graphbudget_residual_norm',wraps=fused.residual_rms_norm) as called:
    output=layer.forward_cuda(x,res);assert called.call_count==0
   assert all(bit_equal(a,b) for a,b in zip(layer.forward_native(x,res),output))
   result['checks'].append({'case':'unknown device mocked fallback'})
   # Compare the actual patched layer against native over both supported widths.
   for n in (896,1536):
    check_layer=layernorm.RMSNorm(n,eps=1e-6).to(device='cuda',dtype=torch.bfloat16)
    check_layer.weight.data.copy_(torch.randn_like(check_layer.weight))
    for m in (1,2,4,8,12,16):
     for seed in range(10):
      xx=torch.randn((m,n),device='cuda',dtype=torch.bfloat16)
      rr=torch.randn_like(xx)
      if seed%5==0:xx=xx*16;rr=rr/16
      a=check_layer.forward_native(xx,rr);b=check_layer.forward_cuda(xx,rr)
      assert all(bit_equal(v,w) for v,w in zip(a,b)),(n,m,seed)
    result['checks'].append({'case':'random bitwise native comparison','N':n,'trials':60})
    all_x=torch.randn((17,n),device='cuda',dtype=torch.bfloat16);all_r=torch.randn_like(all_x)
    whole=check_layer.forward_native(all_x,all_r)
    for m in (1,2,4,8,12,16,17):
     part=check_layer.forward_cuda(all_x[:m],all_r[:m])
     assert all(bit_equal(ref[:m],got) for ref,got in zip(whole,part)),(n,m,'batch')
     order=torch.arange(m-1,-1,-1,device='cuda')
     reverse=check_layer.forward_cuda(all_x[:m][order],all_r[:m][order])
     assert all(bit_equal(ref[order],got) for ref,got in zip(part,reverse)),(n,m,'reorder')
    result['checks'].append({'case':'batch and row invariance','N':n,'sizes':[1,2,4,8,12,16,17]})
   # A contiguous view may still have a nonzero storage offset.
   xb=torch.randn((8*896+5,),device='cuda',dtype=torch.bfloat16)[3:3+8*896].view(8,896)
   rb=torch.randn((8*896+6,),device='cuda',dtype=torch.bfloat16)[4:4+8*896].view(8,896)
   with patch.object(layernorm,'_graphbudget_residual_norm',wraps=fused.residual_rms_norm) as called:
    actual=layer.forward_cuda(xb,rb);assert called.call_count==1
   assert all(bit_equal(a,b) for a,b in zip(layer.forward_native(xb,rb),actual))
   result['checks'].append({'case':'contiguous nonzero storage offsets'})
   for value in (0.,-0.,1e-20,1e20,3e38,float('inf'),float('-inf'),float('nan')):
    xx=torch.randn((8,896),device='cuda',dtype=torch.bfloat16);rr=torch.randn_like(xx)
    xx[0,1]=value;rr[1,2]=value
    expected=layer.forward_native(xx,rr);actual=layer.forward_cuda(xx,rr)
    assert all(bit_equal(a,b) for a,b in zip(expected,actual)),str(value)
   result['checks'].append({'case':'special values bitwise','values':8})
   model=os.environ.get('GRAPHBUDGET_MODEL_PATH')
   if model:
    from safetensors import safe_open
    with safe_open(str(Path(model)/'model.safetensors'),framework='pt',device='cpu') as f:
     for key in ('model.layers.0.input_layernorm.weight','model.layers.0.post_attention_layernorm.weight'):
      weight=f.get_tensor(key).to('cuda');n=weight.numel()
      real_layer=layernorm.RMSNorm(n,eps=1e-6).to(device='cuda',dtype=torch.bfloat16)
      real_layer.weight.data.copy_(weight)
      xx=torch.randn((16,n),device='cuda',dtype=torch.bfloat16);rr=torch.randn_like(xx)
      expected=real_layer.forward_native(xx,rr);actual=real_layer.forward_cuda(xx,rr)
      assert all(bit_equal(a,b) for a,b in zip(expected,actual)),key
      result['checks'].append({'case':'real model weight','name':key,'width':n})
   else:
    result['checks'].append({'case':'real model weight','status':'SKIPPED: GRAPHBUDGET_MODEL_PATH unset'})
   os.environ['GRAPHBUDGET_RESIDUAL_NORM']='0'
   with patch.object(layernorm,'_graphbudget_residual_norm',wraps=fused.residual_rms_norm) as called:
    layer.forward_cuda(x,res);assert called.call_count==1
   result['checks'].append({'case':'environment change after import does not alter selected mode'})
   with torch.enable_grad():
    x0=torch.randn((8,896),device='cuda',dtype=torch.bfloat16,requires_grad=True)
    r0=torch.randn((8,896),device='cuda',dtype=torch.bfloat16,requires_grad=True)
    layer.zero_grad(set_to_none=True)
    layer.forward_native(x0,r0)[0].float().sum().backward()
    expected=(x0.grad.clone(),r0.grad.clone(),layer.weight.grad.clone())
    x1=x0.detach().clone().requires_grad_();r1=r0.detach().clone().requires_grad_();layer.zero_grad(set_to_none=True)
    with patch.object(layernorm,'_graphbudget_residual_norm',wraps=fused.residual_rms_norm) as called:
     layer.forward_cuda(x1,r1)[0].float().sum().backward();assert called.call_count==0
    assert all(bit_equal(a,b) for a,b in zip(expected,(x1.grad,r1.grad,layer.weight.grad)))
   result['checks'].append({'case':'grad-enabled native fallback and backward equivalence'})
   x=torch.randn((16,896),device='cuda',dtype=torch.bfloat16);res=torch.randn_like(x)
   layer.forward_cuda(x,res);torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
   with torch.cuda.graph(g):y,rr=layer.forward_cuda(x,res)
   for i in range(4):
    x.copy_(torch.randn_like(x));res.copy_(torch.randn_like(res));g.replay();torch.cuda.synchronize();ref=layer.forward_native(x,res)
    assert bit_equal(y,ref[0]) and bit_equal(rr,ref[1]),i
   result['checks'].append({'case':'captured integration replay','replays':4})
 result['status']='PASS';write_json(out/'result.json',result);print('PASS',len(result['checks']),'patched dispatch/fallback/freeze/replay checks')
except Exception as e:
 result['status']='FAIL';result['error']=f'{type(e).__name__}: {e}';write_json(out/'result.json',result);raise
