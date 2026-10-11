"""Independent held-out teacher inference. Its artifacts never enter training."""
import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'working/MoRGS')]
from phase05.common import SCENES,protocol,read,write,sha
from scene.utils import get_flow_model,get_flow

def fb(f,b):
    _,_,h,w=f.shape
    yy,xx=torch.meshgrid(torch.arange(h,device=f.device),torch.arange(w,device=f.device),indexing='ij')
    x=xx[None]+f[:,0];y=yy[None]+f[:,1]
    warped=F.grid_sample(b,torch.stack((2*x/(w-1)-1,2*y/(h-1)-1),-1),align_corners=True)
    a=f*2;bref=warped*2
    finite=torch.isfinite(f).all(1)&torch.isfinite(warped).all(1)
    return ((x>=0)&(x<=w-1)&(y>=0)&(y<=h-1)&finite&((a+bref).square().sum(1)<=.01*(a.square().sum(1)+bref.square().sum(1))+.5))[0]

def masks(flow,valid,radius=3):
    dynamic=valid&(flow[0]*2).norm(dim=0).gt(1.5)
    d=dynamic.float()[None,None];ks=radius*2+1
    dilate=F.max_pool2d(d,ks,1,radius)>0
    erode=1-F.max_pool2d(1-d,ks,1,radius)
    boundary=(dilate[0,0]&(erode[0,0]<1))&valid
    return dict(dynamic=dynamic,boundary=boundary,static=valid&~dynamic,teacher_inconsistent=~valid)

def main():
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');args=p.parse_args()
    protocol();manifest=ROOT/'manifests/phase05_regions.json'
    if manifest.exists() and not args.smoke:
        d=read(manifest);assert all(sha(ROOT/r['path'])==r['sha256'] for r in d['files']);print('FROZEN_REGIONS_REUSED');return
    torch.manual_seed(0);torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
    ckpt=ROOT/'weights/Tartan-C-T-TSKH-spring540x960-M.pth';cfg=ROOT/'working/MoRGS/RAFT/config/eval/spring-M.json'
    model=get_flow_model(str(ckpt),str(cfg));files=[]
    for scene in SCENES:
        folder=ROOT/'data/dynerf/dev/validation'/scene/'cam01'
        def image(i):
            a=np.asarray(Image.open(folder/f'{i:04d}.png').convert('RGB')).copy()
            return torch.from_numpy(a).permute(2,0,1).float().div(255).cuda()
        count=1 if args.smoke else 89
        for i in range(count):
            with torch.no_grad():
                a,b=image(i),image(i+1)
                forward=get_flow(str(ckpt),str(cfg),model,a,b)[0]
                backward=get_flow(str(ckpt),str(cfg),model,b,a)[0]
                forward[:,1]*=507/506;backward[:,1]*=507/506
                assert forward.shape==(1,2,507,676) and torch.isfinite(forward).all() and torch.isfinite(backward).all()
                entries=[(i,i+1,forward,fb(forward,backward))]
                if i==count-1:entries.append((i+1,i,backward,fb(backward,forward)))
                for frame,target,f,valid in entries:
                    path=ROOT/('diagnostics/phase05/smoke_regions' if args.smoke else 'cache/phase05_eval_regions/dev')/scene/f'{frame:04d}.npz'
                    path.parent.mkdir(parents=True,exist_ok=True)
                    arrays={k:v.cpu().numpy() for k,v in masks(f,valid).items()}
                    np.savez_compressed(path,**arrays)
                    files.append(dict(scene=scene,frame=frame,target=target,path=str(path.relative_to(ROOT)),sha256=sha(path),
                                      pixels={k:int(v.sum()) for k,v in arrays.items()},rgb_sha256=sha(folder/f'{frame:04d}.png')))
            if i%10==0:print('REGIONS_PROGRESS',scene,i,flush=True)
    result=dict(status='SMOKE_PASS' if args.smoke else 'FROZEN',files=files,
                teacher_sha256=sha(ckpt),teacher_config_sha256=sha(cfg),teacher_code_sha256=sha(ROOT/'working/MoRGS/scene/utils.py'),
                protocol_sha256=sha(ROOT/'manifests/phase05_protocol.json'),seconds=time.perf_counter()-start,
                peak_allocated=torch.cuda.max_memory_allocated(),scope='cam01 only, evaluation-only teacher proxy; no test camera access or use by training')
    write(ROOT/'manifests/phase05_regions_smoke.json' if args.smoke else manifest,result)
    print(json.dumps(dict(status=result['status'],frames=len(files),seconds=result['seconds'])))

if __name__=='__main__':main()
