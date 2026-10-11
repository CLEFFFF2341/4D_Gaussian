"""Separate, frozen image evaluation; never imported by the training process."""
import argparse
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'working/MoRGS'))
import torch
import torchvision
from PIL import Image
from torchvision.transforms.functional import to_tensor
from gaussian_renderer import render_mask
from scene.dataset_readers import readCamerasFromPoseBounds
from utils.camera_utils import loadSequentialCam
from utils.loss_utils import ssim
from utils.image_utils import psnr as author_psnr, save_image
from lpipsPyTorch.modules.lpips import LPIPS
from scene import GaussianModel

def psnr(prediction, truth):
    return -10 * torch.log10((prediction - truth).square().mean())


class RenderSnapshot:
    def __init__(self, snapshot):
        assert snapshot['format'] == 'phase03_decoded_render_v1'
        self.a = {k: v.cuda() for k, v in snapshot['attributes'].items()}
        self.frame_idx = snapshot['frame']
        self.active_sh_degree = snapshot['active_sh_degree']
        self.max_sh_degree = snapshot['max_sh_degree']
        self.get_resi = None if snapshot['residual'] is None else snapshot['residual'].cuda()
        self.mask_all = torch.ones((self.a['xyz'].shape[0], 8), dtype=torch.bool, device='cuda')
    @property
    def get_xyz(self): return self.a['xyz']
    @property
    def get_flow(self): return self.a['flow']
    @property
    def get_opacity(self): return torch.sigmoid(self.a['op'])
    @property
    def get_motion_label(self): return torch.sigmoid(self.a['motion_label'])
    @property
    def get_scaling(self): return torch.exp(self.a['sc'])
    @property
    def get_rotation(self): return torch.nn.functional.normalize(self.a['rot'])
    @property
    def get_features(self): return torch.cat((self.a['f_dc'], self.a['f_rest']), dim=1)


def self_test(criterion):
    generator = torch.Generator().manual_seed(717)
    image = torch.rand((1,3,64,64), generator=generator).cuda()
    zeros = torch.zeros_like(image)
    constant = torch.full_like(image, 0.1)
    with torch.no_grad():
        ident = dict(psnr=float(psnr(image,image).mean()), ssim=float(ssim(image,image)),
                     lpips=float(criterion(image*2-1,image*2-1)))
        known = float(psnr(constant,zeros).mean())
        ab = float(criterion(image*2-1,zeros*2-1))
        ba = float(criterion(zeros*2-1,image*2-1))
    assert math.isinf(ident['psnr']) and abs(ident['ssim']-1)<1e-6 and abs(ident['lpips'])<1e-9
    assert abs(known-20)<1e-5 and abs(ab-ba)<1e-7, (known, ab, ba)
    quantized_known = float(author_psnr(constant,zeros))
    assert abs(quantized_known - 20*math.log10(255/26))<1e-5
    result = dict(status='PASS', identical=dict(psnr='infinity',ssim=ident['ssim'],lpips=ident['lpips']),
                  constant_error_psnr=known, author_quantized_constant_psnr=quantized_known,
                  lpips_symmetry_error=abs(ab-ba))
    (ROOT/'manifests/phase03_eval_selftest.json').write_text(json.dumps(result,indent=2),encoding='utf-8')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path)
    parser.add_argument('--protocol',choices=['dev','final'],default='dev')
    parser.add_argument('--scene',default='coffee_martini')
    parser.add_argument('--role',choices=['validation','test'],default='validation')
    parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32=True
    torch.backends.cudnn.allow_tf32=True
    torch.cuda.reset_peak_memory_stats()
    criterion=LPIPS(net_type='vgg',version='0.1').cuda().eval()
    self_test(criterion)
    protocol=dict(rgb='sRGB release PNG, RGB [0,1], full frame, no crop or resizing',
                  render='FP32 decoded Gaussian attributes, RGB clipped [0,1]',
                  psnr='-10 log10 mean RGB MSE; arithmetic mean of per-frame dB',
                  author_psnr='20 log10(255 / sqrt(mean((round(pred*255)-round(gt*255))^2)))',
                  ssim='MoRGS utils.loss_utils.ssim, 11x11 Gaussian sigma1.5, zero padding, C1=.01^2 C2=.03^2',
                  unified_lpips='vendored lpipsPyTorch VGG16 version0.1; explicit input RGB*2-1',
                  author_lpips='same vendored network, author metrics_video.py inputs [0,1]',
                  namespaces=['unified_fp32','author_png8'],
                  torch_version=torch.__version__,torchvision_version=torchvision.__version__,
                  tf32_matmul=True,tf32_cudnn=True,
                  selection='No test-driven selection; validation cam01, final test cam00')
    (ROOT/'eval_protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    if args.self_test:return
    assert args.run
    data=ROOT/'data/dynerf'/args.protocol/args.role/args.scene
    meta=json.loads((data/'dataset_meta.json').read_text(encoding='utf-8'))
    infos,_=readCamerasFromPoseBounds(str(data/'poses_bounds.npy'),meta['size_wh'],N_video_views=10)
    cameras=[loadSequentialCam(SimpleNamespace(data_device='cuda',znear=5.,zfar=300.),i,info,1.)
             for i,info in enumerate(infos)]
    pipe=SimpleNamespace(debug=False,compute_cov3D_python=False,convert_SHs_python=False)
    background=torch.zeros(3,device='cuda')
    records=[]
    start=time.perf_counter()
    render_times=[]
    full_render_times=[]
    full_render_errors=[]
    output=args.run/'evaluation'/args.role
    output.mkdir(parents=True,exist_ok=True)
    for name in meta['cameras']:
        (output/'renders'/name).mkdir(parents=True,exist_ok=True)
    for file in sorted((args.run/'snapshots').glob('*.pt')):
        snapshot=torch.load(file,weights_only=False)
        model=RenderSnapshot(snapshot)
        frame=snapshot['frame']
        checkpoint_file=args.run/'checkpoints'/f'frame_{frame:04d}.pt'
        if checkpoint_file.exists():
            checkpoint=torch.load(checkpoint_file,weights_only=False)
            state=checkpoint['gaussian_state']
            full=GaussianModel(state['max_sh_degree'],state['latent_args'],state['model_args'])
            full.__dict__.update(state);full.setup_functions()
            with torch.no_grad():
                reference=render_mask(cameras[0],model,pipe,background)['render']
                for iteration in range(6):
                    torch.cuda.synchronize();tic=time.perf_counter()
                    native=render_mask(cameras[0],full,pipe,background)['render']
                    torch.cuda.synchronize()
                    if iteration:full_render_times.append(time.perf_counter()-tic)
                error=float((reference-native).abs().max())
                assert error<1e-6,(frame,error)
                full_render_errors.append(dict(frame=frame,abs_max=error))
            del full,checkpoint,state,native,reference
        for camera,name in zip(cameras,meta['cameras']):
            with Image.open(data/name/f'{frame-1:04d}.png') as image:
                gt=to_tensor(image.convert('RGB')).cuda()[None]
            with torch.no_grad():
                torch.cuda.synchronize()
                tic=time.perf_counter()
                prediction=render_mask(camera,model,pipe,background)['render'].clamp(0,1)[None]
                torch.cuda.synchronize()
                render_times.append(time.perf_counter()-tic)
                png=torch.round(prediction*255)/255
                values={}
                for namespace,pred,lpinput in [('unified_fp32',prediction,True),('author_png8',png,False)]:
                    values[namespace]=dict(PSNR=float(psnr(pred,gt) if lpinput else author_psnr(pred,gt)),SSIM=float(ssim(pred,gt)),
                        LPIPS=float(criterion(pred*2-1 if lpinput else pred,gt*2-1 if lpinput else gt)))
                assert all(math.isfinite(v) for group in values.values() for v in group.values())
            save_image(prediction[0],str(output/'renders'/name/f'{frame-1:04d}.png'))
            records.append(dict(frame=frame,source_frame=frame-1,camera=name,metrics=values))
        del model,snapshot,prediction,gt,png
        (output/'per_frame.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
    summary={namespace:{metric:sum(r['metrics'][namespace][metric] for r in records)/len(records)
                       for metric in ['PSNR','SSIM','LPIPS']} for namespace in protocol['namespaces']}
    summary.update(frames=len(records),role=args.role,seconds=time.perf_counter()-start,
                   rendering_fps=len(render_times)/sum(render_times),
                   rendering_fps_definition='one synchronized decoded-snapshot render per evaluated image; loading excluded',
                   full_model_rendering_fps=len(full_render_times)/sum(full_render_times),
                   full_model_fps_definition='original GaussianModel including decoders; 1 warmup +5 timed renders per retained checkpoint, loading excluded',
                   full_model_snapshot_checks=full_render_errors,
                   peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved())
    (output/'results.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
