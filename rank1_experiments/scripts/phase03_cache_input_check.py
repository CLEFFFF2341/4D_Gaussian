import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
result={}
for scene in ['coffee_martini','cut_roasted_beef']:
    old=ROOT/'cache/teachers/dev'/scene
    new=ROOT/'cache/teachers/dev_phase03'/scene
    changed=0;initial_changed=0;maximum=0.;records=0
    for path in sorted(new.glob('*/*.npz')):
        before=old/path.relative_to(new)
        with np.load(path) as current,np.load(before) as previous:
            if path.name.startswith('sam_'):
                difference=int((current['b0_union']!=previous['b0_union']).sum())
                changed+=difference
                if path.name=='sam_0001.npz':initial_changed+=difference
            else:
                maximum=max(maximum,float(np.max(np.abs(current['b0_flow']-previous['b0_flow']))))
                records+=1
    result[scene]=dict(flow_files=records,input_path_flow_abs_max=maximum,
                       sam_union_changed_pixels=changed,initial_motion_mask_changed_pixels=initial_changed,
                       old_cache_preserved=True)
(ROOT/'manifests/phase03_cache_input_precision.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
