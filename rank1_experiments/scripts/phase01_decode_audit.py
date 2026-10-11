"""Strict sequential video decoding; preserve raw input on every failure."""
import concurrent.futures
import argparse
import hashlib
import json
import subprocess
import time
import zipfile
from pathlib import Path
import imageio_ffmpeg

ROOT=Path(__file__).resolve().parents[1]
FFMPEG=imageio_ffmpeg.get_ffmpeg_exe()

def check(video):
    command=[FFMPEG,'-v','error','-xerror','-threads','2','-i',str(video),'-frames:v','300','-f','null','NUL']
    started=time.perf_counter()
    try:
        process=subprocess.run(command,capture_output=True,text=True,timeout=90)
        result=dict(path=str(video),command=command,exit_code=process.returncode,stderr=process.stderr,
                    strict_decode_passed=process.returncode==0,seconds=time.perf_counter()-started)
    except subprocess.TimeoutExpired:
        result=dict(path=str(video),command=command,strict_decode_passed=False,error='90s timeout')
    print(video.parent.name,video.name,result['strict_decode_passed'],flush=True)
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--ffmpeg',default=FFMPEG)
    args=parser.parse_args()
    FFMPEG=args.ffmpeg
    videos=[p for name in ['coffee_martini','cut_roasted_beef'] for p in sorted((Path('E:/4DGS/data')/name).glob('cam*.mp4'))]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        results=list(pool.map(check,videos))
    archive=Path('E:/4DGS/downloads/n3v-all/coffee_martini.zip')
    with zipfile.ZipFile(archive) as z:
        entry=next(n for n in z.namelist() if n.endswith('/cam01.mp4'))
        archived_hash=hashlib.sha256(z.read(entry)).hexdigest()
    disk_hash=hashlib.sha256(Path('E:/4DGS/data/coffee_martini/cam01.mp4').read_bytes()).hexdigest()
    result=dict(ffmpeg=FFMPEG, scope='First 300 frames, all 38 development-camera videos; no RGB saved',
                results=results,coffee_cam01_archive_hash=archived_hash,coffee_cam01_disk_hash=disk_hash,
                coffee_cam01_matches_archive=archived_hash==disk_hash,
                caveat='Strict failure is decoder evidence, not a root-cause diagnosis. Source bitstreams and decoder versions require separate checks.')
    (ROOT/'manifests/decode_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
