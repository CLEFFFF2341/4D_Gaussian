"""Read-only teacher replay; caller must explicitly choose B0 or shared protocol."""
import json
from pathlib import Path
import numpy as np
import torch

class TeacherCache:
    def __init__(self,root,scene,allowed_cameras,profile,max_steps=90):
        if profile not in ['b0','shared']:raise ValueError('Explicit teacher profile required')
        if set(allowed_cameras)&{'cam00','cam01'}:raise ValueError('Held-out view requested')
        self.root=Path(root);self.scene=scene;self.allowed=set(allowed_cameras);self.profile=profile
        self.max_steps=max_steps
    def pair(self,camera,source,target,device='cpu'):
        if camera not in self.allowed:raise ValueError('Camera outside frozen flow subset')
        p=self.root/self.scene/camera/f'{source:04d}_{target:04d}.npz'
        with np.load(p) as a:
            meta=json.loads(str(a['metadata']))
            assert (meta['camera'],meta['source_frame'],meta['target_frame'])==(camera,source,target)
            if self.profile=='b0':
                flow=torch.from_numpy(a['b0_flow'].copy()).to(device)
                valid=torch.ones(flow.shape[-2:],dtype=torch.bool,device=device)
            else:
                flow=torch.from_numpy(a['shared_flow'].astype(np.float32)).to(device)
                valid=torch.from_numpy(a['shared_valid'].copy()).to(device)
            heatmap=torch.from_numpy(a['b0_heatmap'].copy()).to(device)
        return dict(flow=flow,valid=valid,heatmap=heatmap,metadata=meta)
    def b0_step(self,camera,step,device='cpu'):
        if self.profile!='b0':raise ValueError('Original B0 replay requires B0 profile')
        if step<1 or step>self.max_steps:raise ValueError('Step outside explicit frozen cache extent')
        source,target=(0,3) if step==1 else (step-2,step-1)
        result=self.pair(camera,source,target,device)
        refresh=step if step in [1,2] else max(2,(step//10)*10)
        p=self.root/self.scene/camera/f'sam_{refresh:04d}.npz'
        with np.load(p) as a:result['motion_mask']=torch.from_numpy(a['b0_union'].copy()).to(device)
        result['mask_refresh_step']=refresh
        return result
