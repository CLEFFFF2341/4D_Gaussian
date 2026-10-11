"""Audit the complete formal cache extent and recorded online/roundtrip probes."""
import json
import hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
root=ROOT/'cache/teachers/final/coffee_martini'
record=json.loads((ROOT/'manifests/phase03_teacher_final.json').read_text(encoding='utf-8'))
assert record['status']=='PASS' and record['formal300_cache']=='PASS'
flows=[];masks=[];probes=[]
index={}
def record_artifact(header,item):
    archive=Path(item['path'])
    index[str(archive.relative_to(ROOT))]=dict(bytes=archive.stat().st_size,
        archive_sha256_at_generation=item['sha256'],header_sha256=hashlib.sha256(header.read_bytes()).hexdigest())
for camera in ['cam02','cam06','cam10','cam16']:
    pairs=[(i,i+1) for i in range(299)]+[(0,3)]
    pairs=pairs+[(b,a) for a,b in pairs]
    for a,b in pairs:
        path=root/camera/f'{a:04d}_{b:04d}.json'
        item=json.loads(path.read_text(encoding='utf-8'))
        assert (item['camera'],item['source_frame'],item['target_frame'])==(camera,a,b)
        assert item['shape_hw']==[1014,1352] and item['b0_precision']=='float32 lossless'
        assert Path(item['path']).is_file() and Path(item['path']).stat().st_size>0
        record_artifact(path,item)
        assert 0<=item['shared_coverage']<=1 and 0<=item['motion_coverage']<=1
        if 'online_repeat_abs_max' in item:
            assert item['online_repeat_abs_max']==0 and item['cache_roundtrip_abs_max']==0
            probes.append(dict(camera=camera,source=a,target=b,online_abs_max=0,roundtrip_abs_max=0))
        flows.append(item)
    for step in [1,2]+list(range(10,301,10)):
        path=root/camera/f'sam_{step:04d}.json'
        item=json.loads(path.read_text(encoding='utf-8'))
        assert item['camera']==camera and item['training_step']==step and item['shape_hw']==[1014,1352]
        assert 0<=item['sam_coverage']<=item['union_coverage']<=1
        assert Path(item['path']).is_file() and Path(item['path']).stat().st_size>0
        record_artifact(path,item)
        masks.append(item)
assert len(flows)==2400 and len(masks)==128 and len(probes)==12
assert record['sam_online_checks']['coffee_martini']==[0,0,0,0]
result=dict(status='PASS',flow_directions=len(flows),sam_refresh_masks=len(masks),online_flow_probes=probes,
            cache_artifact_index=index,
            sam_initial_repeat_changed_pixels=record['sam_online_checks']['coffee_martini'],
            scope='Full metadata/extent audit, plus generator online and lossless disk probes; not a new inference run or full archive decompression.')
(ROOT/'manifests/phase03_cache_metadata_check.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(dict(status='PASS',flow_directions=len(flows),sam_refresh_masks=len(masks),online_probes=len(probes))))
