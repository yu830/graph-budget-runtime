"""Independent Python integer oracle + complete native metadata-entry microbench."""
import argparse
import fcntl
import itertools
import json
import os
from pathlib import Path
import random
import statistics
import time

import torch
from graphbudget.metadata import configure, supported
from graphbudget.experiment import gpu_lease
from graphbudget.runtime import write_json
from sglang.srt.layers.attention.triton_backend import TritonAttnBackend
from sglang.srt.model_executor.forward_batch_info import ForwardMode

def backend(mode):
    b=TritonAttnBackend.__new__(TritonAttnBackend)
    b.graphbudget_metadata_enabled=True
    b.enable_deterministic=True; b.split_tile_size=256; b.sliding_window_size=None
    b.max_context_len=2048; b.device='cuda'; b.static_kv_splits=False
    b.device_core_count=48; b.max_kv_splits=8
    # Nontrivial pool stride and nontrivial mapping. Never treat pool IDs as KV IDs.
    b.req_to_token=(torch.arange(32*2112,device='cuda',dtype=torch.int32)*3+7).reshape(32,2112)
    b.kv_indptr=torch.zeros(33,dtype=torch.int32,device='cuda')
    b.cuda_graph_num_kv_splits=torch.zeros(32,dtype=torch.int32,device='cuda')
    b.cuda_graph_kv_indices=torch.zeros(32*2048,dtype=torch.int64,device='cuda')
    os.environ['GRAPHBUDGET_METADATA_MODE']=mode; configure(b,32)
    return b

def call(b,seq,pool):
    n=seq.numel()
    b.init_forward_metadata_replay_cuda_graph(n,pool,seq,0,None,ForwardMode.DECODE,None,None)

def validate():
    results=[]; rng=random.Random(20261027)
    for n in (1,3,4,5,8,9,16):
        for pattern in ('ones','zero_boundary','tile_boundary','ragged','max_context','padded'):
            lengths={'ones':[1]*n,'zero_boundary':[0,1,255,256,257,511,512,513]*2,
                     'tile_boundary':[255,256,257,511,512,513,1023,1024]*2,
                     'ragged':[rng.randint(1,1024) for _ in range(n)],
                     'max_context':[2048]*n,'padded':[64]*(max(1,n-2))+[1]*2}[pattern][:n]
            for stride in (1,):
                seqbase=torch.full((n*stride+3,),-777,dtype=torch.int32,device='cuda')
                seq=seqbase[1:1+n*stride:stride];seq.copy_(torch.tensor(lengths,device='cuda',dtype=torch.int32))
                pool_list=list(reversed(range(1,n+1)))
                poolbase=torch.full((2*n+1,),-1,device='cuda',dtype=torch.int32)
                pool=poolbase[1::2];pool.copy_(torch.tensor(pool_list,device='cuda',dtype=torch.int32))
                expected_ptr=[0,*itertools.accumulate(lengths)]
                expected_split=[(x+255)//256 for x in lengths]
                expected_idx=[(r*2112+j)*3+7 for r,l in zip(pool_list,lengths) for j in range(l)]
                for mode in ('native','prealloc','prefix_only','fused'):
                    b=backend(mode)
                    # The upstream gather assumes contiguous req_pool_indices.
                    # Its valid runtime contract is preserved, rather than inventing stride support.
                    pools=pool.contiguous()
                    call(b,seq,pools);torch.cuda.synchronize()
                    got=[b.kv_indptr[:n+1].tolist(),b.cuda_graph_num_kv_splits[:n].tolist(),b.cuda_graph_kv_indices[:sum(lengths)].tolist()]
                    expected=[expected_ptr,expected_split,expected_idx]
                    for field,g,e in zip(('kv_indptr','num_kv_splits','kv_indices'),got,expected):
                        if g!=e:
                            k=next(i for i,(a,z) in enumerate(zip(g,e)) if a!=z)
                            raise AssertionError({'field':field,'index':k,'actual':g[k],'expected':e[k], 'n':n,'pattern':pattern,'mode':mode,'stride':stride})
                    results.append({'batch':n,'pattern':pattern,'mode':mode,'seq_stride':seq.stride(),
                                    'seq_storage_offset':seq.storage_offset(),'pool_stride':pools.stride(),
                                    'dtype':'int32','device':'cuda','status':'PASS'})
    # Scope guards: no launch in unsupported shape/settings.
    b=backend('fused');s=torch.ones(32,device='cuda',dtype=torch.int32)
    assert not supported(b,4,s[::2],b.kv_indptr,b.cuda_graph_num_kv_splits)
    for n in (0,17,32): assert not supported(b,n,s,b.kv_indptr,b.cuda_graph_num_kv_splits)
    b.enable_deterministic=False;assert not supported(b,4,s,b.kv_indptr,b.cuda_graph_num_kv_splits)
    b.enable_deterministic=True;b.sliding_window_size=256;assert not supported(b,4,s,b.kv_indptr,b.cuda_graph_num_kv_splits)
    return results

def measure():
    samples=[];rng=random.Random(19)
    for n,length in ((1,64),(4,128),(3,256),(8,128),(9,256),(16,64)):
        seq=torch.full((n,),length,device='cuda',dtype=torch.int32)
        pools=torch.arange(n,device='cuda',dtype=torch.int32)
        backends={m:backend(m) for m in ('native','prealloc','prefix_only','fused')}
        for b in backends.values():
            for _ in range(30):call(b,seq,pools)
        for pair in range(10):
            order=list(backends);rng.shuffle(order)
            # Two same-code native blocks estimate within-round A/A noise.
            order=['native',*order,'native']
            for pos,m in enumerate(order):
                torch.cuda.synchronize();start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                start.record();t=time.perf_counter_ns()
                for _ in range(300):call(backends[m],seq,pools)
                end.record();end.synchronize();wall=(time.perf_counter_ns()-t)/300/1000
                samples.append({'batch':n,'context':length,'round':pair,'position':pos,'mode':m,
                                'complete_call_wall_us':wall,'event_span_us':start.elapsed_time(end)*1000/300})
    return samples

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--correctness-only',action='store_true');a=p.parse_args();out=Path(a.output)
    out.mkdir(parents=True,exist_ok=False)
    with open(os.environ['GRAPHBUDGET_SHARED_GPU_LOCK'],'a') as lock,gpu_lease():
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        results=validate();write_json(out/'correctness.json',{'oracle':'Python int accumulate/ceildiv + explicit pool storage formula','cases':results})
        print('exact cases',len(results),flush=True)
        if a.correctness_only:return
        samples=measure();write_json(out/'raw.json',{'timing':'no profiler; whole metadata method; 300 calls per sample; synchronization only at sample boundaries','samples':samples})
        for n in (1,3,4,8,9,16):
            print(n,{m:round(statistics.median(r['complete_call_wall_us'] for r in samples if r['batch']==n and r['mode']==m),2) for m in ('native','prealloc','prefix_only','fused')},flush=True)

if __name__=='__main__':main()
