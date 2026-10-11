"""Run pinned pretrained teachers on a training-camera frame pair."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / 'working/MoRGS'
sys.path.insert(0, str(WORK))
os.environ['TORCH_HOME'] = str(ROOT / 'cache/torch')
import cv2
import numpy as np
import torch

def frames():
    video = Path('E:/4DGS/data/coffee_martini/cam02.mp4')
    capture = cv2.VideoCapture(str(video))
    result = []
    for i in range(2):
        ok, frame = capture.read()
        assert ok, i
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result.append(cv2.resize(rgb, (676, 507), interpolation=cv2.INTER_AREA))
    capture.release()
    return result

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def convert():
    from safetensors.torch import load_file
    from RAFT.core.raft import RAFT
    from RAFT.config.parser import json_to_args
    src = ROOT / 'weights/sea_raft_model.safetensors'
    dst = ROOT / 'weights/Tartan-C-T-TSKH-spring540x960-M.pth'
    args = json_to_args(str(WORK / 'RAFT/config/eval/spring-M.json'))
    model = RAFT(args)
    tensors = load_file(str(src), device='cpu')
    original_keys=set(tensors)
    aliases={}
    # Safetensors omits duplicate names for the same BatchNorm module.
    # The vendored model registers bn3 again as downsample[1].
    for name,value in model.state_dict().items():
        if name in tensors:
            continue
        assert '.downsample.1.' in name, name
        alias=name.replace('.downsample.1.','.bn3.')
        assert alias in tensors and tensors[alias].shape==value.shape, name
        prefix=name.split('.downsample.1.')[0]
        block=model.get_submodule(prefix)
        assert block.bn3 is block.downsample[1]
        tensors[name]=tensors[alias]
        aliases[name]=alias
    result = model.load_state_dict(tensors, strict=True)
    assert not result.missing_keys and not result.unexpected_keys
    torch.save(tensors, dst)
    restored = torch.load(dst, map_location='cpu', weights_only=True)
    assert tensors.keys() == restored.keys()
    assert all(torch.equal(tensors[k], restored[k]) for k in tensors)
    return dict(status='PASS', parameter_keys=len(tensors),
                original_safetensors_keys=len(original_keys),shared_module_aliases=aliases,
                tensors_bitwise_preserved=True, strict_load=True,
                safetensors_sha256=sha(src), pth_sha256=sha(dst),
                hf_repository='MemorySlices/Tartan-C-T-TSKH-spring540x960-M',
                hf_revision='eb97ef34ba5d856c3fa2cdcd073150c057ac8b69')

def raft():
    from scene.utils import get_flow_model, get_flow
    checkpoint = ROOT / 'weights/Tartan-C-T-TSKH-spring540x960-M.pth'
    config = WORK / 'RAFT/config/eval/spring-M.json'
    images = [torch.from_numpy(x.copy()).permute(2,0,1).float().cuda()/255 for x in frames()]
    model = get_flow_model(str(checkpoint), str(config))
    assert not model.training
    outputs = []
    with torch.no_grad():
        for a, b in [(0, 1), (1, 0)]:
            flow, info, heatmap = get_flow(str(checkpoint), str(config), model, images[a], images[b])
            torch.cuda.synchronize()
            assert flow.shape == (1,2,507,676), flow.shape
            assert all(torch.isfinite(x).all() for x in [flow,info,heatmap])
            outputs.append(dict(direction=f'{a}->{b}', shape=list(flow.shape),
                                abs_mean=flow.abs().mean().item(), abs_max=flow.abs().max().item()))
        flow, _, _ = get_flow(str(checkpoint), str(config), model, images[0], images[0])
        assert torch.isfinite(flow).all()
        identity_mean=flow.norm(dim=1).mean().item()
    np.savez_compressed(ROOT/'manifests/raft_smoke_arrays.npz',
                        identity_flow=flow.cpu().numpy())
    return dict(status='PASS', eval_mode=True, no_grad=True, checkpoint_sha256=sha(checkpoint),
                calls=outputs, identity_mean_epe_pixels=identity_mean,
                input_range='[0,1], matching vendored MoRGS call path; no preprocessing correction',
                caveat='Finite outputs and strict weights establish execution, not flow quality or cache agreement.')

def sam():
    from scene.utils import get_sam2_model, get_sam2_mask
    from sam2.utils.misc import get_connected_components
    binary = torch.zeros((1,1,8,10),dtype=torch.uint8,device='cuda')
    binary[0,0,1:3,1:3] = 1
    binary[0,0,5:7,6:9] = 1
    labels, counts = get_connected_components(binary)
    torch.cuda.synchronize()
    assert counts[0,0,1,1].item()==4 and counts[0,0,5,6].item()==6
    assert labels[0,0,1,1]!=labels[0,0,5,6]
    model = get_sam2_model(str(ROOT/'weights/sam2.1_hiera_large.pt'),
                          'configs/sam2.1/sam2.1_hiera_l.yaml')
    assert not model.training
    images = [torch.from_numpy(x.copy()).permute(2,0,1).float().cuda()/255 for x in frames()]
    prompt = torch.zeros((507,676),dtype=torch.bool,device='cuda')
    prompt[180:300,250:420] = True
    with torch.no_grad():
        outputs = get_sam2_mask(model, images, [prompt])
    torch.cuda.synchronize()
    assert len(outputs)==2
    assert all(x.shape==(507,676) for x in outputs)
    np.savez_compressed(ROOT/'manifests/sam2_smoke_arrays.npz',
                        masks=np.stack([x.cpu().numpy() for x in outputs]))
    return dict(status='PASS', connected_components_cuda=True, component_areas=[4,6],
                init_state3=True, mask_prompt=True, propagated_frames=len(outputs), eval_mode=True,
                coverage=[x.float().mean().item() for x in outputs],
                checkpoint_sha256=sha(ROOT/'weights/sam2.1_hiera_large.pt'),
                precision='float32; TF32 enabled by upstream helper',
                caveat='Execution health only; the synthetic rectangle is not a motion teacher mask.')

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=['convert','raft','sam'])
    args=parser.parse_args()
    torch.manual_seed(0)
    torch.cuda.reset_peak_memory_stats()
    started=time.perf_counter()
    result=globals()[args.mode]()
    result.update(torch=torch.__version__,cuda=torch.version.cuda,
                  python=sys.version.split()[0], seconds=time.perf_counter()-started,
                  peak_allocated=torch.cuda.max_memory_allocated(),
                  peak_reserved=torch.cuda.max_memory_reserved(),
                  camera='coffee_martini/cam02',frame_ids=[0,1],
                  official_environment=True)
    path=ROOT/'manifests'/f'teacher_{args.mode}_smoke.json'
    path.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
