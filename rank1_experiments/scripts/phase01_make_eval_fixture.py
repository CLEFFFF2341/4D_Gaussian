"""Prepare synthetic images to exercise the existing evaluation file contract."""
import json
from pathlib import Path
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'smoke/metrics_contract'
for kind in ['gt','renders']:
    (OUT/'test'/kind/'synthetic_camera').mkdir(parents=True,exist_ok=True)
y,x=np.mgrid[:128,:128]
for frame in range(2):
    a=np.stack([(x+y+frame)%200+20,(2*x+y)%200+20,(x+3*y)%200+20],axis=-1).astype(np.uint8)
    Image.fromarray(a).save(OUT/'test/gt/synthetic_camera'/f'{frame:04d}.png')
    Image.fromarray(a+1).save(OUT/'test/renders/synthetic_camera'/f'{frame:04d}.png')
(OUT/'fixture.json').write_text(json.dumps(dict(synthetic=True,baseline_results=False,
    expected_psnr_db=48.1308036087,frames=2,camera='synthetic_camera'),indent=2),encoding='utf-8')
print(str(OUT))
