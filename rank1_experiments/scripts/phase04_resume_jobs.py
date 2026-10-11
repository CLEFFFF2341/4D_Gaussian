"""Two independent continuations from the same M boundary for variability audit."""
import json
import subprocess
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
base = ROOT/'runs/phase04/native/coffee_martini/M'
assert (base/'resource.json').exists()
for name in ('resume', 'repeat'):
    run = ROOT/'runs/phase04/resume/coffee_martini'/name
    if (run/'resource.json').exists(): continue
    command = [sys.executable, str(ROOT/'scripts/phase04_launch.py'), 'M_'+name,
               str(ROOT/'working/MoRGS/train_phase04.py'), '--config',
               str(ROOT/'configs/phase04/native_coffee_martini_M.yaml'), '-m', str(run),
               '--start_checkpoint', str(base/'checkpoints/frame_0002.pt'), '--stop_after_frame', '4',
               '--flow_method', 'M', '--flow_k', '64', '--flow_grad_ratio', '.1']
    assert subprocess.run(command, cwd=ROOT.parent).returncode == 0
