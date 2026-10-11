import json
import shutil
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
source=ROOT/'runs/B0/smoke/coffee_martini/continuous_v2'
target=ROOT/'runs/B0/dev/coffee_martini/seed0'
for sub,name in [('snapshots','0001.pt'),('checkpoints','frame_0001.pt')]:
    destination=target/sub/name
    destination.parent.mkdir(parents=True,exist_ok=True)
    assert not destination.exists(),destination
    shutil.copy2(source/sub/name,destination)
first=json.loads((source/'resources.jsonl').read_text(encoding='utf-8').splitlines()[0])
first['reused_from']=str(source)
(target/'resources.jsonl').write_text(json.dumps(first)+'\n',encoding='utf-8')
(target/'initial_reuse.json').write_text(json.dumps(dict(source=str(source),
    explanation='Same seed0, train-only init cloud, all 16 train views, 676x507, initial10000; extend sequence extent after frame1.',
    original_initial_seconds=first['train_and_save_seconds']),indent=2),encoding='utf-8')
print(str(target/'checkpoints/frame_0001.pt'))
