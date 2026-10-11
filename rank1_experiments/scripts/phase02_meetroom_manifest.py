"""Inspect public StreamRF Drive metadata without downloading scene videos."""
import json
from html.parser import HTMLParser
from pathlib import Path
import re
import subprocess
import numpy as np
from phase02_prepare_data import ROOT,save,sha
FOLDERS={'discussion':'1ZT7WU_giCBDezj_q21zlaiFhQOWBY_gD','trimming':'1EtOJ1Mo3lKCmsV_A1TFTnhcIBbNPEPVw','vrheadset':'17fX5zp-t311K4xtuJU6H1a2WRydKPOvZ'}
class Entries(HTMLParser):
    def __init__(self):super().__init__();self.entries=[]
    def handle_starttag(self,tag,attrs):
        a=dict(attrs)
        if tag=='div' and a.get('class')=='JxSEve' and 'aria-label' in a:
            m=re.search(r'5:auSv138:([\w-]+)-0-16',a.get('ssk',''))
            if m:self.entries.append(dict(name=a['aria-label'].split(' ')[0],drive_id=m.group(1)))
def fetch(url,p):
    r=subprocess.run(['curl.exe','-sS','--proxy','http://127.0.0.1:7897','--ssl-revoke-best-effort',
       '--connect-timeout','10','--max-time','25','-L',url,'-o',str(p)],capture_output=True,text=True)
    assert r.returncode==0,r.stderr
result=dict(status='MANIFEST_ONLY',official_entry='https://github.com/AlgoHunt/StreamRF#dataset',
   official_drive='https://drive.google.com/drive/folders/1lNmQ6_ykyKjT6UKy-SnqWoSlI5yjh3l_',
   paper='https://arxiv.org/abs/2603.25042',expected_protocol=dict(scenes=3,cameras=13,frames=300,fps=30,held_out='central view'),
   modelscope=dict(tree_reachable=True,revision='8422e507d5a802b45b4fdba99ff31b8daa7bb722',
       file_download='API returned Code10020301002 login required; no account credentials requested',
       extra_scene='stepin exists in ModelScope listing; excluded from original three-folder Drive release'),scenes=[])
directory=ROOT/'sources/meetroom_metadata'
for scene,folder in FOLDERS.items():
    html=directory/f'{scene}_drive.html';parser=Entries();parser.feed(html.read_text(encoding='utf-8'))
    entries=parser.entries;cams=[x['name'] for x in entries if x['name'].endswith('.mp4')]
    assert len(cams)==13,(scene,entries)
    pose=next(x for x in entries if x['name']=='poses_bounds.npy');p=directory/f'{scene}_poses_bounds.npy'
    if not p.exists():fetch('https://drive.usercontent.google.com/download?id='+pose['drive_id']+'&export=download',p)
    row=dict(scene=scene,drive_folder=folder,entries=entries,cameras=cams,videos_downloaded=False,
       listing_sha256=sha(html),poses_downloaded=False,test_camera=None)
    try:
        a=np.load(p);assert a.shape==(13,17) and np.isfinite(a).all()
        row.update(poses_downloaded=True,poses_sha256=sha(p),poses_path=str(p),poses_shape=list(a.shape),
           raw_hwf=a[0,:15].reshape(3,5)[:,4].tolist(),
           test_camera_candidate='cam_0',test_camera_status='StreamRF llffhold100 selects first image; awaiting explicit MoRGS/data central-camera cross-check')
    except Exception as e:row['pose_error']=str(e)
    result['scenes'].append(row)
save(ROOT/'manifests/meetroom_manifest.json',result)
for file,url in [('streamrf_llff.py','https://raw.githubusercontent.com/AlgoHunt/StreamRF/main/util/llff_dataset.py'),
                 ('streamrf_meetroom.json','https://raw.githubusercontent.com/AlgoHunt/StreamRF/main/configs/meetroom.json'),
                 ('streamrf_prepare_dataset.py','https://raw.githubusercontent.com/AlgoHunt/StreamRF/main/prepare_dataset.py')]:
    fetch(url,directory/file)
print(json.dumps(result,indent=2))
