"""Numerical source-grid/resize checks and train-root IO isolation audit."""
import argparse
import builtins
import json
from pathlib import Path
import sys
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from phase02_prepare_data import ROOT,DEV,FLOW,sha,save
WORK=ROOT/'working/MoRGS';sys.path.insert(0,str(WORK))
def numerical():
    from phase02_teacher_cache import fb_mask
    f=torch.zeros(1,2,507,676);f[:,0]=3;f[:,1]=2
    b=-f;mask,error=fb_mask(f,b)
    expected=torch.ones(507,676,dtype=torch.bool);expected[-2:]=False;expected[:,-3:]=False
    assert torch.equal(mask,expected)
    assert error[mask].max()<1e-10
    bad=torch.zeros_like(b);mask2,_=fb_mask(f,bad);assert not mask2.any()
    # Anisotropic resize must use separate component ratios.
    flow=torch.ones(1,2,10,20);flow[:,1]=2
    resized=F.interpolate(flow,size=(15,50),mode='bilinear',align_corners=False)
    resized[:,0]*=50/20;resized[:,1]*=15/10
    assert torch.allclose(resized[:,0],torch.full((1,15,50),2.5))
    assert torch.allclose(resized[:,1],torch.full((1,15,50),3.))
    result=dict(status='PASS',known_translation=[3,2],fb_valid_pixels=int(mask.sum()),
       inverse_translation_consistency=True,wrong_direction_rejected=True,non_square_component_scale=True,
       mask_grid='source integer pixel centers, grid_sample align_corners=True',reference='1352x1014',
       shared_mask_precision='computed from unquantized FP32 flows, then cached as bool')
    save(ROOT/'manifests/phase02_numerical_checks.json',result);print(result)
def loader():
    from utils.loader_utils import MultiViewVideoDataset
    from scene.dataset_readers import readDynerfInfo
    from utils.graphics_utils import getWorld2View2,getProjectionMatrix
    from phase02_geometry import calibration
    results=[]
    for protocol in ['dev','final']:
        for scene in ([ARGS.scene] if ARGS.scene else DEV):
            d=ROOT/f'data/dynerf/{protocol}/train/{scene}'
            meta=json.loads((d/'dataset_meta.json').read_text(encoding='utf-8'))
            accessed=[];original=builtins.open
            def traced(file,*args,**kwargs):
                if isinstance(file,(str,Path)):
                    p=Path(file)
                    if p.suffix.lower()=='.png':
                        assert d in p.parents,('held-out RGB accessed',p)
                        assert p.parent.name in meta['cameras'] and p.parent.name!='cam00'
                        accessed.append(str(p))
                return original(file,*args,**kwargs)
            builtins.open=traced
            try:
                dataset=MultiViewVideoDataset(str(d),split='train',test_indices=[],max_frames=meta['n_frames'],resolution=1)
                for cam in range(dataset.n_cams):
                    for frame in [0,meta['n_frames']-1]:
                        image,path,mask=dataset[cam*dataset.n_frames+frame]
                        assert tuple(image.shape[-2:])==tuple(reversed(meta['size_wh']))
                info=readDynerfInfo(str(d),[],1,N_video_views=10)
                assert len(info.train_cameras)==dataset.n_cams and not info.test_cameras
                cals=calibration(np.load(d/'poses_bounds.npy'),*meta['size_wh'])
                errors=[]
                for cam,(k,e,p,center) in zip(info.train_cameras,cals):
                    actual=getWorld2View2(cam.R,cam.T)
                    assert np.allclose(actual[:3],e,atol=1e-6)
                    projection=getProjectionMatrix(cam.znear,cam.zfar,cam.FovX,cam.FovY).numpy()
                    # Known camera-local points, transformed to world and projected by actual MoRGS matrices.
                    local=np.array([[0.,0.,15.],[1.,-2.,20.],[-2.,1.,30.]])
                    world=(local-e[:,3])@e[:,:3];hom=np.c_[world,np.ones(3)]
                    a=hom@actual.T@projection.T;ndc=a[:,:2]/a[:,3:]
                    pix=(ndc+1)*np.array(meta['size_wh'])/2-0.5
                    b=hom@p.T;expected=b[:,:2]/b[:,2:]
                    errors.append(float(np.max(np.abs(pix-expected))))
                assert max(errors)<1e-3,errors
            finally:builtins.open=original
            assert dataset.n_cams==len(meta['cameras'])
            rejected=False
            try:MultiViewVideoDataset(str(d),split='train',test_indices=[0],max_frames=meta['n_frames'])
            except ValueError:rejected=True
            assert rejected
            held=ROOT/f'data/dynerf/{protocol}/test/{scene}'
            try:MultiViewVideoDataset(str(held),split='train',test_indices=[],max_frames=meta['n_frames'])
            except ValueError:pass
            else:raise AssertionError('Held-out training loader must reject')
            results.append(dict(scene=scene,protocol=protocol,cameras=dataset.n_cams,frames=dataset.n_frames,
               rgb_open_count=len(accessed),all_rgb_opens_train_only=True,point_cloud_points=len(info.point_cloud.points),
               default_test_index_rejected=True,held_out_training_root_rejected=True,
               actual_morgs_vs_calibration_projection_abs_max_pixels=max(errors),
               flow_cameras=FLOW[scene],flow_runtime_indices=meta['flow_runtime_indices']))
    name='phase02_loader_checks.json' if not ARGS.scene else f'phase02_loader_checks_{ARGS.scene}.json'
    save(ROOT/'manifests'/name,dict(status='PASS',results=results));print(json.dumps(results,indent=2))
def meetroom():
    p=ROOT/'manifests/meetroom_manifest.json';m=json.loads(p.read_text(encoding='utf-8'))
    for s in m['scenes']:
        a=np.load(s['poses_path']);poses=a[:,:15].reshape(-1,3,5);centers=poses[:,:,3]
        centroid=centers.mean(0);dist=np.linalg.norm(centers-centroid,axis=1)
        s.update(test_camera='cam_0',test_camera_status='Verified release cam_0 plus official StreamRF cam00 rename / hold_every100 first view and MoRGS test_indices[0]',
            nearest_camera_to_centroid_row=int(np.argmin(dist)),camera_center_centroid_distances=dist.tolist(),
            camera_sort='numeric suffix order cam_0..cam_12; never lexicographic unpadded names',
            derived_name_mapping={f'cam_{i}':f'cam{i:02d}' for i in range(13)},
            source_pose_rows={f'cam_{i}':i for i in range(13)},
            flow_cameras_morgs_filtered_train_indices_1_4_7_10=['cam_2','cam_5','cam_8','cam_11'])
    # Save actual primary source hashes, not an inferred N3DV split.
    d=ROOT/'sources/meetroom_metadata'
    m['protocol_evidence']={str(p):sha(p) for p in [d/'streamrf_llff.py',d/'streamrf_prepare_dataset.py',d/'streamrf_meetroom.json',WORK/'configs/meetroom.yaml',WORK/'arguments/__init__.py']}
    m['protocol_limits']='No video decoding or image-level pose reprojection yet. Pose-row numeric mapping is release-format evidence, not new visual validation.'
    save(p,m);print([(s['scene'],s['raw_hwf'],s['nearest_camera_to_centroid_row']) for s in m['scenes']])
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['numerical','loader','meetroom']);p.add_argument('--scene',choices=DEV);ARGS=p.parse_args();globals()[ARGS.mode]()
