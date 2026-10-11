"""Read-only source rasterization/export. Original rasterizer remains installed."""
import math
from pathlib import Path
import torch
from .shared_flow import match_uid, project, flow_loss, resolution_scale

_extension = None


def extension():
    global _extension
    if _extension is None:
        from torch.utils.cpp_extension import load
        _extension = load(name='morgs_phase04_candidates_v1',
                          sources=[str(Path(__file__).with_name('candidates.cu'))],
                          extra_cuda_cflags=['-O2'], verbose=True)
    return _extension


@torch.no_grad()
def snapshot(pc, uid):
    return dict(xyz=pc.get_xyz.detach().clone(), opacity=pc.get_opacity.detach().clone(),
                scaling=pc.get_scaling.detach().clone(), rotation=pc.get_rotation.detach().clone(),
                uid=uid.detach().clone())


@torch.no_grad()
def raster_candidates(source, camera, target_uid, k=8, debug=False):
    from gaussian_rasterization_grad import _C
    xyz = source['xyz']
    mapping = match_uid(source['uid'], target_uid)
    empty = torch.empty(0)  # native forward treats empty CPU arguments as null
    n = len(xyz)
    h, w = camera.image_height, camera.image_width
    if not n:
        return dict(ids=torch.full((h,w,k), -1, dtype=torch.long, device=xyz.device),
                    weights=xyz.new_zeros((h,w,k)), mass=xyz.new_zeros((h,w,2)), mapping=mapping,
                    native_alpha=xyz.new_zeros((1,h,w)), debug=None)
    args = (xyz.new_zeros(3), xyz, empty, xyz.new_zeros((n, 1)),
            xyz.new_zeros((n, 3)), source['opacity'], source['scaling'], source['rotation'],
            1., empty, empty, empty.bool(), empty.bool(), camera.world_view_transform,
            camera.full_proj_transform, math.tan(camera.FoVx/2), math.tan(camera.FoVy/2),
            h, w, xyz.new_zeros((n, 1, 3)), 0, camera.camera_center, False, False, False, True, False, False)
    raw = _C.rasterize_gaussians(*args)
    out = extension().export_candidates(raw[9], raw[10], raw[11], (mapping >= 0).contiguous(),
                                        n, raw[0], h, w, k, debug)
    return dict(ids=out[0], weights=out[1], mass=out[2], mapping=mapping,
                native_alpha=raw[5], debug=out[3:] if debug else None)


def candidate_loss(source, target_xyz, camera, candidates, teacher, valid,
                   method='M', sigma=2., epsilon=.02, chunk=32768):
    h, w = teacher.shape[-2:]
    ids, weights = candidates['ids'].reshape(-1, candidates['ids'].shape[-1]), candidates['weights'].flatten(0, 1)
    source_xy, source_valid = project(source['xyz'], camera.full_proj_transform,
                                      camera.world_view_transform, w, h)
    target_xy, target_valid = project(target_xyz, camera.full_proj_transform,
                                      camera.world_view_transform, w, h)
    mapping = candidates['mapping']
    if len(mapping) and len(target_xyz):
        trows = mapping[ids.clamp_min(0)]
    else:
        # Empty source/target geometry still scores every fixed valid pixel.
        total, diagnostics = flow_loss(target_xyz.new_zeros((h*w,ids.shape[-1],2)),
            weights*0, teacher.permute(1,2,0).reshape(-1,2), valid.flatten(), method, sigma, epsilon,
            scale=resolution_scale(w,h))
        diagnostics.update(uid_mass_retention=0., topk_mass_retention=0., geometry_mass_retention=0.)
        return total + target_xyz.sum()*0, diagnostics
    good = (ids >= 0) & (trows >= 0) & source_valid[ids.clamp_min(0)] & target_valid[trows.clamp_min(0)]
    safe_weights = weights * good
    y = teacher.permute(1, 2, 0).reshape(-1, 2)
    mask = valid.flatten().detach().bool()
    # Camera invalidity of every candidate is an abstention, never removes Ω.
    denominator = (mask & torch.isfinite(y).all(-1)).sum().clamp_min(1)
    total = target_xyz.sum()*0
    stats = []
    for start in range(0, len(ids), chunk):
        sl = slice(start, start+chunk)
        f = target_xy[trows[sl].clamp_min(0)]-source_xy[ids[sl].clamp_min(0)].detach()
        loss, diag = flow_loss(f, safe_weights[sl], y[sl], mask[sl], method, sigma, epsilon,
                               scale=resolution_scale(w, h))
        total = total + loss*diag.get('valid_pixels', 0)/denominator
        stats.append(diag)
    valid_count = sum(d.get('valid_pixels', 0) for d in stats)
    den = max(1, valid_count)
    mass = candidates['mass'].reshape(-1, 2)
    native_den = mass[mask, 0].sum().clamp_min(1e-12)
    matched_den = mass[mask, 1].sum().clamp_min(1e-12)
    diagnostics = dict(method=method, valid_pixels=valid_count,
        candidate_coverage=sum(d.get('candidate_pixels', 0) for d in stats)/den,
        abstain_pixels=sum(d.get('abstain_pixels', 0) for d in stats),
        uid_mass_retention=float(mass[mask, 1].sum()/native_den),
        topk_mass_retention=float(weights[mask].sum()/matched_den),
        geometry_mass_retention=float(safe_weights[mask].sum()/weights[mask].sum().clamp_min(1e-12)),
        outlier_posterior=sum(d.get('outlier_posterior', 0)*d.get('valid_pixels', 0) for d in stats)/den,
        responsibility_entropy=sum(d.get('responsibility_entropy', 0)*d.get('valid_pixels', 0) for d in stats)/den)
    return total, diagnostics
