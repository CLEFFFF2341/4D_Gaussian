"""Extract pinned N3DV frames into physically separated protocol directories."""
import argparse
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time
import numpy as np
from PIL import Image
import cv2

ROOT = Path(__file__).resolve().parents[1]
FFMPEG = ROOT/'sources/ffmpeg-0.6.0/imageio_ffmpeg/binaries/ffmpeg-win-x86_64-v7.1.exe'
DEV = ['coffee_martini', 'cut_roasted_beef']
FLOW = {'coffee_martini':['cam02','cam06','cam10','cam16'],
        'cut_roasted_beef':['cam02','cam06','cam10','cam15']}
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''): h.update(b)
    return h.hexdigest()
def save(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_suffix(p.suffix+'.tmp'); t.write_text(json.dumps(x,indent=2),encoding='utf-8'); t.replace(p)
def init():
    if (ROOT/'manifests/dataset_manifest.json').exists():
        raise FileExistsError('Phase 02 already initialized. Use extract to resume; do not overwrite frozen splits.')
    inventory=json.loads((ROOT/'manifests/dataset_inventory.json').read_text(encoding='utf-8'))
    manifests={}
    for protocol,size,n in [('dev',[676,507],90),('final',[1352,1014],300)]:
        scenes={}
        for s in inventory['scenes']:
            name=s['scene']; cams=[c['camera'] for c in s['cameras']]
            train=[c for c in cams if c!='cam00' and (protocol!='dev' or c!='cam01')]
            flow=FLOW.get(name,[cams[1:][i] for i in [1,4,8,13]])
            scenes[name]=dict(train=train,validation=['cam01'] if protocol=='dev' else [],test=['cam00'],
                              pose_rows={c:i for i,c in enumerate(cams)},flow_cameras=flow,
                              flow_runtime_indices=[train.index(c) for c in flow],frame_range=[0,n-1],size_wh=size)
            if name not in DEV: continue
            raw=Path(s['path']); poses=np.load(raw/'poses_bounds.npy')
            for role,selected in [('train',train),('validation',scenes[name]['validation']),('test',['cam00'])]:
                if not selected: continue
                d=ROOT/f'data/dynerf/{protocol}/{role}/{name}';d.mkdir(parents=True,exist_ok=True)
                np.save(d/'poses_bounds.npy',poses[[cams.index(c) for c in selected]])
                save(d/'dataset_meta.json',dict(protocol=protocol,role=role,cameras=selected,size_wh=size,
                     n_frames=n,source_pose_rows=[cams.index(c) for c in selected],
                     test_indices=[],flow_runtime_indices=[train.index(c) for c in flow] if role=='train' else [],
                     public_calibration_source=str(raw/'poses_bounds.npy'),
                     test_rgb_allowed=role=='test',initialization='train-only frame 0 calibrated SIFT triangulation'))
        manifests[protocol]=dict(protocol=protocol,scenes=scenes,split_frozen_at=datetime.now(timezone(timedelta(hours=8))).isoformat(),
          test_rgb_policy='Only preprocessing integrity checks and separate final evaluation may access test RGB.',
          camera_mapping='Sorted actual original MP4 filenames map to full poses_bounds rows; subset poses preserve that order.',
          resolution_convention=dict(raw_wh=[2704,2028],size_wh=size,resize='FFmpeg area raw->formal x0.5; OpenCV INTER_AREA formal->dev x0.5; no crop or rotation',
            pixel_mapping='x_dst=(x_src+0.5)*sx-0.5; y_dst=(y_src+0.5)*sy-0.5',
            flow_mapping='flow_x*=dst_w/src_w; flow_y*=dst_h/src_h',
            b0_principal_point='MoRGS symmetric FoV with (W-1)/2,(H-1)/2 raster pixel center',
            strict_calibrated_principal_point='raw W/2,H/2 transformed with half-pixel resize; same (dst-1)/2+scale/2'))
        save(ROOT/f'manifests/splits_{protocol}.json',manifests[protocol])
    save(ROOT/'manifests/dataset_manifest.json',dict(status='EXTRACTING',source='https://github.com/facebookresearch/Neural_3D_Video',
         source_inventory=str(ROOT/'manifests/dataset_inventory.json'),scenes=inventory['scenes'],
         prepared_scenes=DEV,extractor_sha256=sha(FFMPEG),extractor=str(FFMPEG),records=[]))
    state=json.loads((ROOT/'experiment_state.json').read_text(encoding='utf-8'))
    state.update(phase='02',status='IN_PROGRESS',phase02='IN_PROGRESS',continue_gate_passed=False,
                 phase02_started_at=datetime.now(timezone(timedelta(hours=8))).isoformat(),
                 phase02_gate=dict(data_complete=False,camera_reprojection=False,teacher_cache=False,split_isolation=False))
    save(ROOT/'experiment_state.json',state)
def extract():
    manifest_path=ROOT/'manifests/dataset_manifest.json';m=json.loads(manifest_path.read_text(encoding='utf-8'))
    for s in m['scenes']:
        if s['scene'] not in DEV: continue
        for c in s['cameras']:
            name=s['scene'];cam=c['camera']; key=name+'/'+cam
            record_path=ROOT/f'manifests/extraction/{name}/{cam}.json'
            if record_path.exists(): print('RESUME',key,flush=True);continue
            started=time.perf_counter(); paths={}
            for protocol,size,n in [('final',[1352,1014],300),('dev',[676,507],90)]:
                role='test' if cam=='cam00' else ('validation' if protocol=='dev' and cam=='cam01' else 'train')
                d=ROOT/f'data/dynerf/{protocol}/{role}/{name}/{cam}';d.mkdir(parents=True,exist_ok=True)
                paths[protocol]=(d,size,n)
            final=paths['final'][0];dev=paths['dev'][0]
            cmd=[str(FFMPEG),'-nostdin','-hide_banner','-loglevel','error','-xerror','-threads','1','-i',c['path'],
                 '-filter_threads','1','-vf','scale=1352:1014:flags=area','-frames:v','300',
                 '-fps_mode','passthrough','-pix_fmt','bgr24','-threads','1','-f','rawvideo','pipe:1']
            log_path=ROOT/f'logs/extract_{name}_{cam}.log'
            with log_path.open('wb') as log:
                proc=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=log)
                try:
                    for i in range(300):
                        byte_count=1352*1014*3; data=bytearray()
                        while len(data)<byte_count:
                            block=proc.stdout.read(byte_count-len(data))
                            if not block: raise RuntimeError(f'{key}: incomplete raw frame {i}')
                            data.extend(block)
                        image=np.frombuffer(data,dtype=np.uint8).reshape(1014,1352,3)
                        assert cv2.imwrite(str(final/f'{i:04d}.png'),image,[cv2.IMWRITE_PNG_COMPRESSION,2])
                        if i<90:
                            small=cv2.resize(image,(676,507),interpolation=cv2.INTER_AREA)
                            assert cv2.imwrite(str(dev/f'{i:04d}.png'),small,[cv2.IMWRITE_PNG_COMPRESSION,2])
                    proc.stdout.close();code=proc.wait()
                    if code: raise RuntimeError(f'{key}: strict decoder exit {code}')
                finally:
                    if proc.poll() is None: proc.kill();proc.wait()
            records={}
            for protocol,(d,size,n) in paths.items():
                imgs=sorted(d.glob('*.png'));assert len(imgs)==n,(key,len(imgs),n)
                rows=[]
                for i,p in enumerate(imgs):
                    assert p.name==f'{i:04d}.png'
                    try:
                        with Image.open(p) as image:
                            image.load();assert list(image.size)==size
                    except Exception as e:
                        raise RuntimeError(f'PNG integrity failure: {p}: {e}') from e
                    rows.append(dict(frame=i,source_video_frame=i,timestamp_seconds=i/30,path=str(p),bytes=p.stat().st_size,sha256=sha(p)))
                records[protocol]=rows
            save(record_path,dict(scene=name,camera=cam,pose_row=c['sorted_pose_row'],video=c['path'],video_sha256=sha(c['path']),
                 command=cmd,frames=records,seconds=time.perf_counter()-started,status='PASS',
                 encoding='OpenCV PNG compression=2; dev from final via INTER_AREA x0.5; every PNG fully decoded by Pillow'))
            print('PASS',key,'seconds',round(time.perf_counter()-started,2),flush=True)
    m['records']=[str(p) for p in sorted((ROOT/'manifests/extraction').glob('*/*.json'))]
    assert len(m['records'])==38
    m.update(status='PASS',development_png_count=38*90,formal_png_count=38*300,
             temporal_alignment='Frame index and 30 fps matched. Content-level synchronization inherited from release; not independently proven.')
    save(manifest_path,m)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['init','extract']);p.add_argument('--max-attempts',type=int,default=3);a=p.parse_args()
    if a.mode=='init':init()
    else:
        for attempt in range(1,a.max_attempts+1):
            try:
                extract();break
            except Exception as e:
                print('FAILED ATTEMPT',attempt,repr(e),flush=True)
                history=ROOT/'manifests/extraction_failures.json'
                previous=json.loads(history.read_text(encoding='utf-8')) if history.exists() else []
                previous.append(dict(time=datetime.now(timezone(timedelta(hours=8))).isoformat(),attempt=attempt,error=repr(e)))
                save(history,previous)
                for log in (ROOT/'logs').glob('extract_*.log'):
                    if log.stat().st_size and '_failed_' not in log.stem:
                        shutil.copy2(log,log.with_name(log.stem+f'_failed_{len(previous)}.log'))
                if attempt==a.max_attempts:raise
