"""Generate a reviewable derivative of the pinned MoRGS training entry point."""
import ast
import difflib
import hashlib
import json
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
source = ROOT / 'working/MoRGS/train.py'
text = source.read_text(encoding='utf-8')
original = text

def replace(old, new):
    global text
    assert text.count(old) == 1, (old[:100], text.count(old))
    text = text.replace(old, new)

replace('EPS = 1.0e-7', '''sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
from phase03_hooks import (CachedTeachers, boundary_save, guard_training_rgb, configure_initial_cloud,
                           load_boundary, restore_boundary, seed_all)
EPS = 1.0e-7''')
replace('import os\n', 'import os\nfrom pathlib import Path\n')
replace('    wandb_enabled = WANDB_FOUND and dataset.use_wandb', '''    guard_training_rgb()
    configure_initial_cloud(args)
    resume = load_boundary(checkpoint)
    wandb_enabled = WANDB_FOUND and dataset.use_wandb''')
replace("    test_image_dataset = MultiViewVideoDataset(dataset.source_path, split='test', test_indices=dataset.test_indices, \n                                               max_frames=dataset.max_frames, start_idx=dataset.start_idx, \n                                               img_format=dataset.img_fmt, resolution=dataset.resolution)", '''    test_image_dataset = Namespace(n_cams=0)''')
text = text.replace('sampler=train_sampler, num_workers=4', 'sampler=train_sampler, num_workers=0')
replace('    train_data = next(train_loader)', '''    train_data = next(train_loader)
    if resume:
        for _ in range(resume['frame'] - 1):
            train_data = next(train_loader)''')
text = text.replace("'mask': test_mask_images.cuda()", "'mask': None")
replace('    flow_model = get_flow_model(args.flow_model_ckpt, args.flow_config)\n    sam2_predictor = get_sam2_model(args.sam2_checkpoint, args.sam2_model_cfg)', '''    cached_teachers = CachedTeachers(args)
    flow_model = None
    sam2_predictor = None
    global get_flow, get_sam2_mask
    get_flow = cached_teachers.flow
    get_sam2_mask = cached_teachers.sam''')
replace('    start_frame_idx = 1', '''    start_frame_idx = 1
    if resume:
        start_frame_idx = restore_boundary(gaussians, train_cameras, resume, generator, args)
        cached_teachers.step = resume['frame']
        cached_teachers.index = resume['frame'] * len(cached_teachers.cameras)
        cur_data = next(train_loader)
        cur_train_images, cur_train_paths, cur_train_masks = cur_data[0].cuda(), cur_data[1], cur_data[2]
        cur_frame_views = cur_train_images
        del resume''')
replace('        first_iter = 1', '''        frame_resource_start = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        first_iter = 1''')
replace('optical_flow_sam2_masks[sam_idx] + optical_flow_masks[sam_idx]', 'optical_flow_sam2_masks[sam_idx]')
replace("test_image_data = {'image':test_images,'path':test_paths, 'mask':test_masks}", "test_image_data = {'image':test_images,'path':test_paths, 'mask':None}")
replace('        # Update previous frame\'s attributes and latents for next frame\'s residual encoding', '''        boundary_save(args, gaussians, train_cameras, generator, frame_idx,
                      frame_resource_start, opt.iterations, ema_loss_for_log)

        # Update previous frame's attributes and latents for next frame's residual encoding''')
# Remove author testing/logging during training. Held-out evaluation is a separate process.
begin = text.index('def training_report(')
end = text.index('if __name__ == "__main__":', begin)
text = text[:begin] + '''def training_report(tb_writer, wandb_enabled, model_args, frame_idx, iteration, Ll1, loss,
                    l1_loss, size, elapsed, is_test, scene, renderFunc, renderArgs, prev_report=None):
    if not is_test:
        return None
    with torch.no_grad():
        camera = scene.getTrainCameras()[0]
        image = renderFunc(camera, scene.gaussians, *renderArgs)['render'].clamp(0, 1)
        return {'val': {'psnr': psnr(image, camera.original_image).mean().double(),
                        'l1': l1_loss(image, camera.original_image).mean().double()},
                'iteration': iteration}

''' + text[end:]
text = text.replace('    safe_state(args.quiet)', '    safe_state(args.quiet)\n    seed_all(args.seed)')
replace("    args = parser.parse_args(sys.argv[1:])", "    parser.add_argument('--stop_after_frame', type=int, default=0)\n    parser.add_argument('--phase03_initial_cloud', type=str, default='')\n    args = parser.parse_args(sys.argv[1:])")
ast.parse(text)
target = ROOT / 'working/MoRGS/train_phase03.py'
target.write_text(text, encoding='utf-8')
patch = ''.join(difflib.unified_diff(original.splitlines(True), text.splitlines(True),
                                   fromfile='train.py (phase02 working)', tofile='train_phase03.py'))
(ROOT / 'manifests/phase03_train_adapter.patch').write_text(patch, encoding='utf-8')
statepath = ROOT / 'experiment_state.json'
state = json.loads(statepath.read_text(encoding='utf-8'))
state.update(phase='03', status='RUNNING', phase03='RUNNING', continue_gate_passed=False,
             phase03_started=datetime.now().astimezone().isoformat(),
             next_phase='03', baseline_reproduction_complete=False)
statepath.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding='utf-8')
print(json.dumps({'adapter': str(target), 'sha256': hashlib.sha256(text.encode()).hexdigest()}))
