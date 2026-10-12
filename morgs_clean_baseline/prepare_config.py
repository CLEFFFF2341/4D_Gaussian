"""Remove unused teacher settings from the independent six-attribute configs."""
from pathlib import Path
import yaml
BASE=Path(__file__).resolve().parent
for path in (BASE/'configs').glob('*.yaml'):
    cfg=yaml.safe_load(path.read_text())
    for section in cfg.values():
        for key in list(section):
            if any(t in key for t in ('flow','motion_label','sam2','keyframe_offset_sam','offset_lr','lambda_offset','frz_offset')):
                del section[key]
    cfg['model_params']['model_path']=str(BASE/'runs'/path.stem)
    cfg['model_params'].update(morgs_flow_supervision=False,morgs_motion_offset=False,morgs_motion_confidence=False)
    path.write_text(yaml.safe_dump(cfg,sort_keys=False),encoding='utf-8')
