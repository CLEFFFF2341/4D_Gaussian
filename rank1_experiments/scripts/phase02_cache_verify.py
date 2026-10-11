"""Verify frozen caches, replay all B0 schedule entries, and plot real training pairs."""
import json
import sys
import numpy as np
import torch
from PIL import Image
from phase02_prepare_data import ROOT,DEV,FLOW,sha,save
from protocol_cache import TeacherCache
sys.path.insert(0,str(ROOT/'working/MoRGS'))
def run():
    m=json.loads((ROOT/'manifests/teacher_manifest.json').read_text(encoding='utf-8'))
    assert m['status']=='PASS' and m['teacher_gpu_unloaded']
    assert len(m['flow_records'])==1440 and len(m['sam_records'])==88
    assert sha(m['models']['raft'])==m['models']['raft_sha256']
    assert sha(m['models']['sam2'])==m['models']['sam2_sha256']
    assert sha(m['models']['raft_config'])==m['models']['config_sha256']
    flow=[];sam=[];checks=[]
    for p in m['flow_records']:
        row=json.loads(open(p,encoding='utf-8').read());assert sha(row['path'])==row['sha256']
        assert row['checkpoint_sha256']==m['models']['raft_sha256']
        assert row['config_sha256']==m['models']['config_sha256']
        with np.load(row['path']) as a:
            b0=a['b0_flow'];shared=a['shared_flow'];valid=a['shared_valid'];heat=a['b0_heatmap']
            assert b0.shape==(2,507,676) and b0.dtype==np.float32
            assert shared.dtype==np.float16 and valid.dtype==np.bool_
            assert np.isfinite(b0).all() and np.isfinite(shared).all()
            assert np.isfinite(heat).all()
            error=np.linalg.norm(shared.astype(np.float32)-b0*np.array([1,507/506],dtype=np.float32)[:,None,None],axis=0)
            assert abs(float(error.mean())-row['fp16_mean_epe'])<1e-5
            assert row['camera'] in FLOW[row['scene']] and row['source_frame']<=89 and row['target_frame']<=89
        flow.append(row)
        if 'online_repeat_abs_max' in row:
            assert row['cache_roundtrip_abs_max']==0 and row['online_repeat_abs_max']<1e-4
            checks.append(row)
    assert len(checks)==24
    for p in m['sam_records']:
        row=json.loads(open(p,encoding='utf-8').read());assert sha(row['path'])==row['sha256']
        assert row['checkpoint_sha256']==m['models']['sam2_sha256']
        assert row['config_sha256']==sha(ROOT/'working/MoRGS/sam2/configs/sam2.1/sam2.1_hiera_l.yaml')
        with np.load(row['path']) as a:assert a['b0_union'].dtype==np.bool_ and a['b0_union'].shape==(507,676)
        sam.append(row)
    assert all(x==0 for vals in m['sam_online_checks'].values() for x in vals)
    replay=[]
    for scene in DEV:
        cache=TeacherCache(ROOT/'cache/teachers/dev',scene,FLOW[scene],'b0')
        shared=TeacherCache(ROOT/'cache/teachers/dev',scene,FLOW[scene],'shared')
        for cam in FLOW[scene]:
            for step in range(1,91):
                a=cache.b0_step(cam,step);assert a['valid'].all() and a['motion_mask'].dtype==torch.bool
            for step,expected in [(1,1),(2,2),(9,2),(10,10),(19,10),(90,90)]:
                assert cache.b0_step(cam,step)['mask_refresh_step']==expected
            replay.append(dict(scene=scene,camera=cam,steps=90,status='PASS'))
        try:cache.pair('cam00',0,1)
        except ValueError:pass
        else:raise AssertionError('Test RGB teacher access allowed')
        # A real frame pair, deterministic selection independent of test results.
        cam=FLOW[scene][0];a,b=0,3
        original=cache.pair(cam,a,b);masked=shared.pair(cam,a,b)
        import matplotlib;matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from RAFT.core.utils.flow_viz import flow_to_image
        source=np.array(Image.open(ROOT/f'data/dynerf/dev/train/{scene}/{cam}/{a:04d}.png'))
        target=np.array(Image.open(ROOT/f'data/dynerf/dev/train/{scene}/{cam}/{b:04d}.png'))
        f=masked['flow'].permute(1,2,0).numpy();valid=masked['valid'].numpy()
        mask=cache.b0_step(cam,1)['motion_mask'].numpy()
        fig,axs=plt.subplots(2,3,figsize=(15,8))
        for ax,img,title in zip(axs.flat,[source,target,flow_to_image(f),valid,mask,source],
            [f'{scene}/{cam} source 0','target 3','Forward flow: x right / y down',
             f'Shared FB-valid {valid.mean():.2%}','B0 SAM union, kept separate','Flow vectors x5 for visibility']):
            ax.imshow(img,cmap='gray' if img.ndim==2 else None);ax.set_title(title);ax.axis('off')
        yy,xx=np.mgrid[12:507:24,12:676:24];sample=f[yy,xx]
        axs.flat[-1].quiver(xx,yy,sample[:,:,0]*5,sample[:,:,1]*5,color='lime',angles='xy',scale_units='xy',scale=1)
        fig.tight_layout();p=ROOT/f'figures/phase02/{scene}_teacher_pair.png';p.parent.mkdir(parents=True,exist_ok=True);fig.savefig(p,dpi=120);plt.close(fig)
    summary=dict(status='PASS',flow_cache_files=len(flow),sam_cache_files=len(sam),online_flow_samples=len(checks),
      b0_cache_roundtrip_abs_max=max(x['cache_roundtrip_abs_max'] for x in checks),
      online_flow_repeat_abs_max=max(x['online_repeat_abs_max'] for x in checks),sam_repeat_changed_pixels=m['sam_online_checks'],
      fp16_abs_max=max(x['fp16_abs_max'] for x in flow),fp16_mean_epe=float(np.mean([x['fp16_mean_epe'] for x in flow])),
      shared_mask_coverage_min=float(min(x['shared_coverage'] for x in flow)),
      shared_mask_coverage_mean=float(np.mean([x['shared_coverage'] for x in flow])),
      shared_mask_coverage_max=float(max(x['shared_coverage'] for x in flow)),
      b0_sam_union_coverage_mean=float(np.mean([x['union_coverage'] for x in sam])),
      b0_schedule_replay=replay,b0_schedule_steps_verified=720,teacher_processes_for_replay=0,
      precision_acceptance='FP16 maximum <0.05 dev pixels; mean EPE <0.001 dev pixels; B0 stays lossless FP32')
    assert summary['fp16_abs_max']<0.05 and summary['fp16_mean_epe']<0.001
    save(ROOT/'manifests/teacher_cache_verification.json',summary);print(json.dumps(summary,indent=2))
if __name__=='__main__':run()
