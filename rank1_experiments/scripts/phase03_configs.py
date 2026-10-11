import json
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
base = yaml.safe_load((ROOT/'working/MoRGS/configs/dynerf.yaml').read_text(encoding='utf-8'))
(ROOT/'configs/phase03').mkdir(parents=True,exist_ok=True)
for protocol, frames in [('dev',90),('final',300)]:
    splits=json.loads((ROOT/'manifests'/f'splits_{protocol}.json').read_text())
    for scene, split in splits['scenes'].items():
        config=yaml.safe_load(yaml.safe_dump(base))
        config['model_params'].update(source_path=str(ROOT/'data/dynerf'/protocol/'train'/scene),
            model_path=str(ROOT/'runs/B0'/protocol/scene/'seed0'), resolution=1,
            max_frames=frames, seed=0, test_indices=[], flow_views=split['flow_runtime_indices'],
            timed=True, use_wandb=False, log_images=False, log_ply=False,
            log_compressed=False, test_interval=0)
        config['model_params'].update(flow_model_ckpt=str(ROOT/'weights/Tartan-C-T-TSKH-spring540x960-M.pth'),
            flow_config=str(ROOT/'working/MoRGS/RAFT/config/eval/spring-M.json'),
            sam2_checkpoint=str(ROOT/'weights/sam2.1_hiera_large.pt'),
            sam2_model_cfg='configs/sam2.1/sam2.1_hiera_l.yaml')
        target=ROOT/'configs/phase03'/f'{protocol}_{scene}.yaml'
        target.write_text(yaml.safe_dump(config,sort_keys=False),encoding='utf-8')
        if protocol=='dev' and scene=='coffee_martini':
            config['model_params'].update(max_frames=10,model_path=str(ROOT/'runs/B0/smoke/coffee_martini/continuous'))
            (ROOT/'configs/phase03/smoke_coffee.yaml').write_text(yaml.safe_dump(config,sort_keys=False),encoding='utf-8')
print('Phase03 frozen configs written; author initial 10000 iterations and rest 8 epochs/view retained.')
