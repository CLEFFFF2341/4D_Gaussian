"""Ordinary Gaussian renderer; no material partition or child branches."""
import math, sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parent.parent
sys.path[:0]=[str(Path(__file__).resolve().parent/'native'),str(ROOT/'rank1_experiments/working/MoRGS')]
import clean_rasterization as native
from utils.sh_utils import eval_sh

def render_arrays(camera,x,s,q,o,c,bg):
    cfg=native.GaussianRasterizationSettings(camera.image_height,camera.image_width,
        math.tan(camera.FoVx/2),math.tan(camera.FoVy/2),bg,1.,camera.world_view_transform,
        camera.full_proj_transform,0,camera.camera_center,False,False,False,True,False,False)
    q=torch.nn.functional.normalize(q,dim=-1)
    screen=torch.zeros_like(x,requires_grad=True);screen.retain_grad()
    mask=torch.zeros((len(x),8),dtype=torch.bool,device=x.device);mask[:,:6]=True
    outputs=native.GaussianRasterizer(cfg)(means3D=x,flow3D=torch.zeros_like(x),means2D=screen,
        colors_precomp=c,motion_label=o.new_zeros((len(x),1)),opacities=o,scales=s,rotations=q,
        update_mask=mask,color_mask=torch.ones(len(x),dtype=torch.bool,device=x.device))
    image,motion_image,flow,inf,count,depth,alpha,radii=outputs
    result=dict(render=image,rendered_motion=motion_image,flow=flow,alpha=alpha,depth=depth,
        radii=radii,visibility_filter=radii>0,viewspace_points=screen,influence=inf,count_influence=count,
        colors=c,settings=cfg)
    return result

def render(camera,pc,bg):
    x=pc.get_xyz
    c=torch.clamp_min(eval_sh(pc.active_sh_degree,pc.get_features.transpose(1,2),
        torch.nn.functional.normalize(x-camera.camera_center,dim=-1))+.5,0)
    return render_arrays(camera,x,pc.get_scaling,pc.get_rotation,pc.get_opacity,c,bg)
