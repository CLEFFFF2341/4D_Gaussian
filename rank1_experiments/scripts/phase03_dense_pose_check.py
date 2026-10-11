"""Verify COLMAP triangulation has not changed the frozen public camera geometry."""
import importlib.util
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
source=ROOT/'working/MoRGS/scene/colmap_loader.py'
spec=importlib.util.spec_from_file_location('phase03_colmap_reader',source)
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
work=ROOT/'initialization/colmap_dense/final/coffee_martini'
extrinsics=module.read_extrinsics_binary(str(work/'sparse/images.bin'))
intrinsics=module.read_intrinsics_binary(str(work/'sparse/cameras.bin'))
data=ROOT/'data/dynerf/final/train/coffee_martini'
meta=json.loads((data/'dataset_meta.json').read_text(encoding='utf-8'))
poses=np.load(data/'poses_bounds.npy')[:,:-2].reshape(-1,3,5)
errors=[]
for name,pose in zip(meta['cameras'],poses):
    converted=np.concatenate([pose[:,1:2],-pose[:,0:1],pose[:,2:4]],axis=1)
    R=-converted[:,:3];R[:,0]*=-1
    T=-converted[:,3].dot(R)
    matches=[c for c in extrinsics.values() if c.name==name+'.png']
    assert len(matches)==1,name
    camera=matches[0]
    error=dict(camera=name,rotation_abs_max=float(np.max(np.abs(camera.qvec2rotmat()-R.T))),
               translation_abs_max=float(np.max(np.abs(camera.tvec-T))))
    assert error['rotation_abs_max']<1e-6 and error['translation_abs_max']<1e-5,error
    errors.append(error)
result=dict(status='PASS',cameras=len(errors),errors=errors,
            intrinsics=[dict(id=c.id,model=c.model,width=c.width,height=c.height,params=c.params.tolist()) for c in intrinsics.values()])
(ROOT/'manifests/phase03_dense_pose_check.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
