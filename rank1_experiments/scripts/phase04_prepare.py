"""Generate a common derivative; leave phase03's frozen adapter unmodified."""
import ast
import difflib
import hashlib
import json
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
original = (ROOT/'working/MoRGS/train_phase03.py').read_text(encoding='utf-8')
text = original


def replace(old, new):
    global text
    assert text.count(old) == 1, (old[:100], text.count(old))
    text = text.replace(old, new)


replace('EPS = 1.0e-7', '''sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from phase04.model import SharedGaussianModel as GaussianModel
from phase04.training import SharedTraining
EPS = 1.0e-7''')
replace('    for frame_idx in range(start_frame_idx, n_frames+1):', '''    shared = SharedTraining(args, gaussians)
    for frame_idx in range(start_frame_idx, n_frames+1):
        shared.begin_frame(frame_idx)''')
begin = text.index('            if render_flow==True:')
end = text.index('            # 3D Motion Residual Field Regularization', begin)
text = text[:begin]+'''            if render_flow:
                loss += shared.flow(viewpoint_cam, recon_loss, iteration)

'''+text[end:]
replace('''        boundary_save(args, gaussians, train_cameras, generator, frame_idx,
                      frame_resource_start, opt.iterations, ema_loss_for_log)''', '''        requested_stop = args.stop_after_frame
        args.stop_after_frame = 0
        boundary_save(args, gaussians, train_cameras, generator, frame_idx,
                      frame_resource_start, opt.iterations, ema_loss_for_log)
        args.stop_after_frame = requested_stop
        shared.end_frame(frame_idx)
        if requested_stop == frame_idx:
            print('PHASE04_CONTROLLED_STOP complete checkpoint and diagnostics saved', flush=True)
            raise SystemExit(0)''')
replace("    args = parser.parse_args(sys.argv[1:])", '''    parser.add_argument('--flow_method', choices=['F0','B1','B2','H','M'], default='M')
    parser.add_argument('--flow_k', type=int, default=8)
    parser.add_argument('--flow_sigma', type=float, default=2.)
    parser.add_argument('--flow_epsilon', type=float, default=.02)
    parser.add_argument('--flow_lambda', type=float, default=-1.)
    parser.add_argument('--flow_grad_ratio', type=float, default=.1)
    parser.add_argument('--flow_freeze_topology', action='store_true')
    args = parser.parse_args(sys.argv[1:])''')
ast.parse(text)
target = ROOT/'working/MoRGS/train_phase04.py'
target.write_text(text, encoding='utf-8')
(ROOT/'patches/phase04_train_adapter.patch').write_text(''.join(difflib.unified_diff(
    original.splitlines(True), text.splitlines(True), fromfile='train_phase03.py (frozen)', tofile='train_phase04.py')), encoding='utf-8')
statepath = ROOT/'experiment_state.json'
state = json.loads(statepath.read_text(encoding='utf-8'))
if state.get('phase') == '03':
    (ROOT/'manifests/phase04_start_state.json').write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding='utf-8')
state.update(phase='04', status='RUNNING', phase04='RUNNING', next_phase='04', continue_gate_passed=False,
             phase04_started=datetime.now().astimezone().isoformat(),
             scope='Phase04 common source candidates, B1/B2/H/M/F0 correctness and short training; no pilot quality claim')
statepath.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding='utf-8')
print(json.dumps(dict(entry=str(target), sha256=hashlib.sha256(target.read_bytes()).hexdigest())))
