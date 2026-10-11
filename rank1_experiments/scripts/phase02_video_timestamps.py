"""Audit container PTS without decoding video RGB; include every existing scene."""
import json
import re
import subprocess
import numpy as np
from phase02_prepare_data import ROOT,FFMPEG,sha,save
inventory=json.loads((ROOT/'manifests/dataset_inventory.json').read_text(encoding='utf-8'))
rows=[]
for s in inventory['scenes']:
    for c in s['cameras']:
        cmd=[str(FFMPEG),'-nostdin','-hide_banner','-loglevel','info','-debug_ts','-i',c['path'],
          '-map','0:v:0','-c','copy','-f','null','NUL']
        run=subprocess.run(cmd,capture_output=True,text=True,encoding='utf-8');assert run.returncode==0
        log=ROOT/f'logs/pts_{s["scene"]}_{c["camera"]}.log';log.write_text(run.stderr,encoding='utf-8')
        pts=np.array([float(x) for x in re.findall(r'muxer <- pts:\S+ pts_time:([\d.eE+-]+)',run.stderr)])
        assert len(pts)==c['frames'],(s['scene'],c['camera'],len(pts),c['frames'])
        ordered=np.sort(pts);err=np.abs(ordered-np.arange(len(pts))/30)
        assert err.max()<5.1e-5,(s['scene'],c['camera'],err.max())
        rows.append(dict(scene=s['scene'],camera=c['camera'],video_sha256=sha(c['path']),
           presentation_frames=len(pts),first_pts=float(ordered[0]),last_pts=float(ordered[-1]),
           maximum_pts_grid_error_seconds=float(err.max()),duplicate_pts=bool(len(np.unique(pts))!=len(pts)),
           timestamps_0_299_same_30fps_grid=True,command=cmd,log=str(log)))
    print('PTS PASS',s['scene'],len(s['cameras']),flush=True)
save(ROOT/'manifests/video_timestamp_audit.json',dict(status='PASS',records=rows,
     checks='All container presentation timestamps, no decoder RGB access; cross-camera index/PTS alignment.',
     limit='Cannot independently prove physical content synchronization; official release states synchronization/invalid-stream filtering.'))
