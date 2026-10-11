"""Retain the early version and rerun its B1 entry with the final shared projection."""
import json
import subprocess
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
run = ROOT/'runs/phase04/frozen/coffee_martini/B1'
archive = ROOT/'runs/phase04_versions/v2/frozen_coffee_martini_B1'
if not archive.exists():
    assert (run/'checkpoints/latest.pt').exists()
    workspace_runs = (ROOT/'runs').resolve()
    assert run.resolve().is_relative_to(workspace_runs) and archive.resolve().is_relative_to(workspace_runs)
    archive.parent.mkdir(parents=True, exist_ok=True)
    run.rename(archive)
    provenance=dict(original_path=str(run),retained_path=str(archive),git_commit='59c12ac',
                    reason='Started before exact pinned w+1e-7 projection denominator correction; engineering history excluded from final comparison')
    (ROOT/'manifests/phase04_superseded_runs.json').write_text(json.dumps([provenance],indent=2),encoding='utf-8')
assert not (run/'resources.jsonl').exists(), 'Run already exists; inspect before retry'
command=[sys.executable,str(ROOT/'scripts/phase04_launch.py'),'regression_B1_final',
         str(ROOT/'working/MoRGS/train_phase04.py'),'--config',str(ROOT/'configs/phase04/frozen_coffee_martini_B1.yaml'),
         '--start_checkpoint',str(ROOT/'runs/B0/dev/coffee_martini/seed0_loader_exact/checkpoints/frame_0001.pt'),
         '--stop_after_frame','4','--flow_method','B1','--flow_k','64','--flow_grad_ratio','.1','--flow_freeze_topology']
assert subprocess.run(command,cwd=ROOT.parent).returncode==0
