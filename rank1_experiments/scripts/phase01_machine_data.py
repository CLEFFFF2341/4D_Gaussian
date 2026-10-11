"""Read-only dataset audit and a bounded CUDA health check."""
import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def save(name, value):
    (ROOT / 'manifests' / name).write_text(json.dumps(value, indent=2), encoding='utf-8')

def data_audit():
    import cv2
    import numpy as np
    scenes = []
    for name in ['coffee_martini', 'cut_roasted_beef', 'cook_spinach', 'flame_salmon_1', 'flame_steak', 'sear_steak']:
        folder = Path('E:/4DGS/data') / name
        pose = folder / 'poses_bounds.npy'
        rows = np.load(pose)
        cameras = []
        for index, video in enumerate(sorted(folder.glob('cam*.mp4'))):
            cap = cv2.VideoCapture(str(video))
            meta = dict(camera=video.stem, sorted_pose_row=index, path=str(video), bytes=video.stat().st_size,
                        opened=cap.isOpened(), frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                        fps=cap.get(cv2.CAP_PROP_FPS))
            sampled = {}
            for frame in [0, 89, 299]:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
                ok, image = cap.read()
                sampled[str(frame)] = dict(decoded=bool(ok), shape=list(image.shape) if ok else None)
            cap.release()
            meta['sampled_frames'] = sampled
            image_folder = folder / video.stem / 'images'
            meta['existing_images'] = len(list(image_folder.glob('*'))) if image_folder.exists() else 0
            cameras.append(meta)
        scenes.append(dict(scene=name, path=str(folder), poses_shape=list(rows.shape), poses_finite=bool(np.isfinite(rows).all()),
                           poses_sha256=hashlib.sha256(pose.read_bytes()).hexdigest(), cameras=cameras,
                           camera_pose_count_match=len(cameras)==len(rows),
                           morgs_points_present=(folder/'points3D_downsample2.ply').exists(),
                           sparse_present=(folder/'sparse_').exists()))
    result = dict(time=datetime.now(timezone(timedelta(hours=8))).isoformat(), mode='headers_and_three_frame_decode_per_camera',
                  full_decode=False, reprojection_verified=False, scenes=scenes,
                  meet_room_local_candidates=[str(p) for p in Path('E:/4DGS/data').iterdir() if 'meet' in p.name.lower() or 'room' in p.name.lower()])
    save('dataset_inventory.json', result)
    for scene in scenes:
        print(scene['scene'], len(scene['cameras']), scene['poses_shape'],
              'all_samples_ok=', all(all(f['decoded'] for f in c['sampled_frames'].values()) for c in scene['cameras']))

def cuda_audit():
    import torch
    result = dict(time=datetime.now(timezone(timedelta(hours=8))).isoformat(), executable=sys.executable,
                  python=sys.version, platform=platform.platform(), torch=torch.__version__, torch_cuda=torch.version.cuda,
                  available=torch.cuda.is_available(), smoke_passed=False)
    result['nvidia_smi'] = subprocess.run(['nvidia-smi'], capture_output=True, text=True).stdout
    started = time.perf_counter()
    try:
        assert torch.cuda.is_available()
        torch.manual_seed(0)
        torch.cuda.reset_peak_memory_stats()
        x = torch.arange(4096, dtype=torch.float32, device='cuda').reshape(64, 64).requires_grad_()
        y = (x.sin().square()).mean()
        y.backward()
        torch.cuda.synchronize()
        assert torch.isfinite(y) and torch.isfinite(x.grad).all()
        cpu = torch.arange(4096, dtype=torch.float32).reshape(64,64)
        error = abs(y.item() - cpu.sin().square().mean().item())
        assert error < 1e-5
        result.update(smoke_passed=True, device=torch.cuda.get_device_name(), cpu_error=error,
                      peak_allocated=torch.cuda.max_memory_allocated(), peak_reserved=torch.cuda.max_memory_reserved(),
                      free_total_bytes=list(torch.cuda.mem_get_info()))
    except Exception as exc:
        result['error'] = repr(exc)
    result['seconds'] = time.perf_counter()-started
    save('environment_health.json', result)
    print(json.dumps({k:v for k,v in result.items() if k!='nvidia_smi'}, indent=2))
    if not result['smoke_passed']:
        raise SystemExit(1)

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('mode', choices=['data','cuda'])
    args=parser.parse_args()
    data_audit() if args.mode=='data' else cuda_audit()
