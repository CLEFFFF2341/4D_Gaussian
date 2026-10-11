"""Independent validation of every completed short run using the frozen metric code."""
import subprocess
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
for topology in ('frozen','native'):
    for scene in ('coffee_martini','cut_roasted_beef'):
        for method in ('F0','B1','B2','H','M'):
            run = ROOT/'runs/phase04'/topology/scene/method
            assert (run/'resource.json').exists(), run
            if (run/'evaluation/validation/results.json').exists(): continue
            command = [sys.executable, str(ROOT/'scripts/phase04_launch.py'), f'eval_{topology}_{scene}_{method}',
                       str(ROOT/'scripts/phase04_evaluate.py'), '--run',str(run), '--scene',scene,'--role','validation']
            assert subprocess.run(command,cwd=ROOT.parent).returncode == 0
