"""Small complete-backbone jobs, same initial state/budget/sampler for all methods."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--single', default='', help='topology/scene/method; otherwise all frozen then all native')
    args = parser.parse_args()
    probe = json.loads((ROOT/'manifests/phase04_real_probe.json').read_text())
    assert probe['status'] == 'PASS'
    k = probe['common_k']
    jobs = []
    for topology in ('frozen', 'native'):
        for scene in ('coffee_martini', 'cut_roasted_beef'):
            for method in ('F0', 'B1', 'B2', 'H', 'M'):
                name = f'{topology}/{scene}/{method}'
                if args.single and args.single != name:
                    continue
                run = ROOT/'runs/phase04'/topology/scene/method
                if (run/'resource.json').exists():
                    resource = json.loads((run/'resource.json').read_text())
                    if [r['frame'] for r in resource['frames']] == [2, 3, 4]:
                        print('Already completed '+name, flush=True); continue
                    raise RuntimeError('Partial run exists: choose explicit new path before retry '+name)
                assert not (run/'resources.jsonl').exists(), 'Do not append to a partial run'
                cfg = yaml.safe_load((ROOT/'configs/phase03'/f'dev_{scene}.yaml').read_text())
                cfg['model_params']['model_path'] = str(run)
                # The true dataset extent, camera sampling, RGB and eight epochs
                # remain unchanged; stop only after a complete frame boundary.
                path = ROOT/'configs/phase04'/f'{topology}_{scene}_{method}.yaml'
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8')
                initial = ROOT/'runs/B0/dev'/scene/'seed0_loader_exact/checkpoints/frame_0001.pt'
                command = [sys.executable, str(ROOT/'scripts/phase04_launch.py'), 'short_'+name.replace('/', '_'),
                           str(ROOT/'working/MoRGS/train_phase04.py'), '--config', str(path),
                           '--start_checkpoint', str(initial), '--stop_after_frame', '4',
                           '--flow_method', method, '--flow_k', str(k), '--flow_grad_ratio', '.1']
                if topology == 'frozen': command.append('--flow_freeze_topology')
                jobs.append(dict(name=name, run=str(run), command=command, k=k, initial_checkpoint=str(initial)))
    (ROOT/'manifests/phase04_short_jobs.json').write_text(json.dumps(jobs, indent=2), encoding='utf-8')
    for job in jobs:
        print('SHORT_JOB '+job['name'], flush=True)
        result = subprocess.run(job['command'], cwd=ROOT.parent)
        assert result.returncode == 0, job['name']


if __name__ == '__main__': main()
