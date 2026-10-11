"""Analytic, numerical and native-buffer equivalence tests for shared losses."""
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'working/MoRGS'))
from phase04.shared_flow import Identity, match_uid, project, reference_candidates, flow_loss, resize_flow, resolution_scale
from phase04.native import raster_candidates
from utils.graphics_utils import getProjectionMatrix

rows = []


def record(name, **evidence):
    rows.append(dict(name=name, status='PASS', **evidence))
    print(json.dumps(rows[-1]), flush=True)


def objective(m, method, w=None, y=None):
    if w is None:
        w = m.new_tensor([[.6, .4]])
    if y is None:
        y = m.new_tensor([[1., -.5]])
    return flow_loss(m, w, y, torch.tensor([True], device=m.device), method,
                     reference_double=m.dtype == torch.float64)[0]


def cpu_tests():
    for method in ('B1', 'B2', 'H', 'M'):
        m = torch.tensor([[[.1, .3], [3., -2.]]], dtype=torch.float64, requires_grad=True)
        analytical = torch.autograd.grad(objective(m, method), m)[0]
        numerical = torch.zeros_like(m)
        for index in np.ndindex(tuple(m.shape)):
            plus, minus = m.detach().clone(), m.detach().clone()
            plus[index] += 1e-5
            minus[index] -= 1e-5
            numerical[index] = (objective(plus, method)-objective(minus, method))/2e-5
        error = float((analytical-numerical).abs().max())
        assert error < 1e-8, (method, error)
        assert analytical.abs().sum() > 0
        record('finite_difference_'+method, max_absolute_error=error)
    w = torch.tensor([[.6, .4]], requires_grad=True)
    m = torch.tensor([[[0., 0.], [2., 1.]]], requires_grad=True)
    loss = objective(m, 'M', w)
    gm, gw = torch.autograd.grad(loss, (m, w), allow_unused=True)
    assert gw is None and gm.norm() > 0
    record('prior_stop_gradient', motion_gradient=float(gm.norm()), weight_gradient=None)
    for k in (1, 2, 8):
        for case in ('zero_weights', 'tiny_alpha', 'coincident', 'huge_motion', 'outlier', 'invalid_pixel'):
            m = torch.zeros(1, k, 2, requires_grad=True)
            w = torch.ones(1, k)
            y = torch.zeros(1, 2)
            valid = torch.ones(1, dtype=torch.bool)
            if case == 'zero_weights': w.zero_()
            if case == 'tiny_alpha': w *= 1e-30
            if case == 'huge_motion': m = torch.full((1, k, 2), 1e20, requires_grad=True)
            if case == 'outlier': y += 1e5
            if case == 'invalid_pixel': valid.zero_(); y[0, 0] = float('nan')
            for method in ('B1', 'B2', 'H', 'M'):
                value, _ = flow_loss(m, w, y, valid, method)
                grad = torch.autograd.grad(value, m)[0]
                assert torch.isfinite(value) and torch.isfinite(grad).all(), (case, method)
    record('edge_cases', cases=18, methods=4)
    for method in ('B2', 'M', 'H'):
        one = torch.tensor([[[2., -1.]]])
        repeated = one.repeat(1, 2, 1)
        if method != 'H':
            assert torch.allclose(objective(one, method, torch.ones(1, 1)), objective(repeated, method))
    record('k1_and_coincident_components')
    empty_loss, diag = flow_loss(torch.zeros(2, 2, 2, requires_grad=True), torch.zeros(2, 2),
                                 torch.zeros(2, 2), torch.ones(2, dtype=torch.bool), 'M')
    expected = -math.log(.02)+math.log(2*math.pi*128**2)
    assert abs(float(empty_loss)-expected) < 1e-5 and diag['abstain_pixels'] == 2
    record('abstention_fixed_denominator', nll=float(empty_loss), expected=expected)
    identity = Identity.initial(3)
    original = identity.uid.clone()
    identity.append(original[[1]])
    identity.append(original[[2]].repeat(2))
    identity.select(torch.tensor([5, 1, 4, 0, 3]))
    matches = match_uid(original, identity.uid)
    assert matches.tolist() == [3, 1, -1] and identity.uid.unique().numel() == 5
    assert (identity.uid[identity.parent >= 0] >= 3).all()
    record('clone_split_prune_reorder_uid', mapping=matches.tolist(), uids=identity.uid.tolist())
    # Explicit perspective: x' = fx*x/z, camera row conventions and signs.
    view = torch.eye(4, dtype=torch.float64)
    projection = getProjectionMatrix(.01, 100., math.pi/2, math.pi/3).T.double()
    xyz = torch.tensor([[0., 0., 2.], [.2, .1, 2.]], dtype=torch.float64)
    pixels, valid = project(xyz, projection, view, 40, 20)
    delta = pixels[1]-pixels[0]
    assert torch.allclose(delta, torch.tensor([2., math.sqrt(3)/2], dtype=torch.float64), atol=1e-7)
    assert valid.all() and torch.allclose(pixels[0], torch.tensor([19.5, 9.5], dtype=torch.float64))
    assert torch.allclose(-delta, pixels[0]-pixels[1])
    record('translation_camera_projection_direction_source_grid', forward=delta.tolist())
    rotated = torch.tensor([[0.,0.,1.,0.], [0.,1.,0.,0.], [-1.,0.,0.,0.], [0.,0.,0.,1.]], dtype=torch.float64)
    world = torch.tensor([[2.,0.,.1], [2.,0.,.3]], dtype=torch.float64)
    pixels, valid = project(world, rotated@projection, rotated, 40, 20)
    assert valid.all() and torch.allclose(pixels[1]-pixels[0], torch.tensor([-2.,0.], dtype=torch.float64), atol=1e-7)
    rotated_delta = (pixels[1]-pixels[0]).tolist()
    depth_move = torch.tensor([[.4,.2,2.], [.4,.2,3.]], dtype=torch.float64)
    pixels, _ = project(depth_move, projection, view, 40, 20)
    perspective_delta = pixels[1]-pixels[0]
    assert perspective_delta.norm() > 1  # full projection includes depth-induced xy motion
    record('extrinsics_camera_depth_and_perspective_motion', rotated_delta=rotated_delta,
           depth_translation_exact_delta=perspective_delta.tolist(), linear_vector_xy_for_pure_z=[0.,0.])
    flow = torch.ones(2, 7, 11)
    resized = resize_flow(flow, 13, 17)
    assert torch.allclose(resized[:, 0, 0], torch.tensor([17/11, 13/7]))
    m = torch.tensor([[[1., 2.], [3., -1.]]])
    assert torch.allclose(objective(m, 'M'),
                          flow_loss(m*2, torch.tensor([[.6, .4]]), torch.tensor([[2., -1.]]),
                                    torch.tensor([True]), 'M', scale=2)[0])
    assert abs(resolution_scale(80, 40)-2*resolution_scale(40, 20)) < 1e-10
    record('non_square_resize_and_reference_units')
    # Full-T counterexample: unmatched .9 foreground still attenuates .5 background.
    xy = torch.zeros(2, 2, dtype=torch.float64)
    conic = torch.tensor([[1., 0., 1., .9], [1., 0., 1., .5]], dtype=torch.float64)
    ids, weights, mass = reference_candidates(xy, conic, torch.arange(2), torch.tensor([[0, 2]]),
                                              1, 1, torch.tensor([False, True]), 1)
    assert ids.item() == 1 and abs(weights.item()-.05) < 1e-10
    record('full_transmittance_includes_unmatched', matched_weight=weights.item(), mass=mass.tolist())
    # Rasterize a translated footprint: supervision at the source pixel must
    # keep its identity even after the footprint leaves that target pixel.
    args = (torch.tensor([[10., 0., 10., .8]], dtype=torch.float64), torch.tensor([0]),
            torch.tensor([[0, 1]]), 3, 8, torch.tensor([True]), 1)
    source_ids = reference_candidates(torch.tensor([[1., 1.]], dtype=torch.float64), *args)[0]
    target_ids = reference_candidates(torch.tensor([[5., 1.]], dtype=torch.float64), *args)[0]
    assert source_ids[1, 1, 0] == 0 and target_ids[1, 1, 0] == -1
    record('source_grid_translated_footprint', source_pixel_has_uid=True, target_same_pixel_has_uid=False)
    for k in (0, 1):
        value, diag = flow_loss(torch.zeros(2, k, 2, requires_grad=True), torch.zeros(2, k),
                                torch.zeros(2, 2), torch.ones(2, dtype=torch.bool), 'M')
        assert torch.isfinite(value) and diag['abstain_pixels'] == 2
    for method in ('B1', 'B2', 'H', 'M'):
        value, _ = flow_loss(torch.zeros(2, 2, 2, requires_grad=True), torch.ones(2, 2),
                             torch.zeros(2, 2), torch.zeros(2, dtype=torch.bool), method)
        assert value == 0
    outlier_flow = torch.tensor([[[1e5, -1e5]]], requires_grad=True)
    value, diag = flow_loss(outlier_flow, torch.ones(1, 1), torch.zeros(1, 2), torch.tensor([True]), 'M')
    gradient = torch.autograd.grad(value, outlier_flow)[0]
    assert diag['outlier_posterior'] == 1 and gradient.norm() == 0
    record('outlier_can_swallow_motion_gradient', outlier_posterior=diag['outlier_posterior'], gradient_norm=float(gradient.norm()))
    # RGB motion from an opaque surface is one latent identity; transparent
    # overlapping layers can produce a valid mean and violate this assumption.
    cases = [
        ('static', [[0., 0.], [0., 0.]], [0., 0.], [.8, .2], True),
        ('opposite_foreground', [[12., 0.], [-12., 0.]], [12., 0.], [.6, .4], True),
        ('single_continuous_surface', [[2., 1.], [2., 1.]], [2., 1.], [.5, .5], True),
        ('transparent_mean_counterexample', [[12., 0.], [-12., 0.]], [0., 0.], [.5, .5], False),
        ('reflection_teacher_counterexample', [[8., 0.], [-4., 0.]], [20., 0.], [.5, .5], False),
    ]
    synthetic = []
    for name, candidates, observed, prior, assumption in cases:
        m = torch.tensor([candidates])
        w = torch.tensor([prior])
        y = torch.tensor([observed])
        values = {method:float(objective(m, method, w, y)) for method in ('B1', 'B2', 'H', 'M')}
        synthetic.append(dict(case=name, single_latent_assumption=assumption, losses=values,
                              candidates=candidates, teacher=observed, prior=prior))
    assert synthetic[1]['losses']['M'] < synthetic[1]['losses']['B2']
    assert synthetic[3]['losses']['B2'] < synthetic[3]['losses']['M']
    disappear = match_uid(torch.tensor([10, 20]), torch.tensor([20, 30]))
    assert disappear.tolist() == [-1, 0]
    record('disappearance_and_disocclusion', mapping=disappear.tolist(), newborn_uid=30)
    record('analytic_assumption_and_counterexamples', cases=synthetic)
    return synthetic


def gpu_tests():
    torch.backends.cuda.matmul.allow_tf32 = False
    h, w = 11, 19
    camera = SimpleNamespace(image_height=h, image_width=w, FoVx=math.pi/2, FoVy=math.pi/3,
                             world_view_transform=torch.eye(4, device='cuda'),
                             full_proj_transform=getProjectionMatrix(.01, 100., math.pi/2, math.pi/3).T.cuda(),
                             camera_center=torch.zeros(3, device='cuda'))
    n = 12
    xyz = torch.zeros(n, 3, device='cuda')
    xyz[:, 2] = torch.linspace(1., 3., n, device='cuda')
    xyz[:, 0] = torch.linspace(-.3, .3, n, device='cuda')
    source = dict(xyz=xyz, opacity=torch.full((n, 1), .15, device='cuda'),
                  scaling=torch.full((n, 3), .3, device='cuda'),
                  rotation=torch.tensor([[1., 0., 0., 0.]], device='cuda').repeat(n, 1),
                  uid=torch.arange(n, device='cuda'))
    for k in (1, 8, 32, 64):
        for mode in ('all', 'pruned', 'none'):
            target = source['uid'] if mode == 'all' else source['uid'][::2] if mode == 'pruned' else torch.zeros(0, device='cuda', dtype=torch.long)
            out = raster_candidates(source, camera, target, k, debug=True)
            xy, conic, point_list, ranges = [v.cpu() for v in out['debug']]
            ref = reference_candidates(xy.double(), conic.double(), point_list, ranges, h, w, out['mapping'].cpu()>=0, k)
            error = float((out['weights'].cpu()-ref[1]).abs().max())
            mass_error = float((out['mass'].cpu()-ref[2]).abs().max())
            assert torch.equal(out['ids'].cpu(), ref[0]) and error < 3e-7 and mass_error < 3e-7
            alpha_error = float((out['mass'][..., 0]-out['native_alpha'].squeeze()).abs().max())
            assert alpha_error < 3e-7
            record('native_cpu_candidates_'+str(k)+'_'+mode, weight_max_error=error,
                   mass_max_error=mass_error, native_alpha_max_error=alpha_error)
    for method in ('B1', 'B2', 'H', 'M'):
        cpu = torch.tensor([[[.1, .3], [3., -2.]]], requires_grad=True)
        gpu = cpu.detach().cuda().requires_grad_()
        lc, lg = objective(cpu, method), objective(gpu, method)
        gc, gg = torch.autograd.grad(lc, cpu)[0], torch.autograd.grad(lg, gpu)[0]
        assert torch.allclose(lc, lg.cpu(), atol=2e-6) and torch.allclose(gc, gg.cpu(), atol=2e-6)
        record('gpu_cpu_loss_gradient_'+method, output_error=float(abs(lc-lg.cpu())), gradient_error=float((gc-gg.cpu()).abs().max()))
    record('gpu', device_name=torch.cuda.get_device_name(0))


if __name__ == '__main__':
    synthetic = cpu_tests()
    if '--cpu-only' not in sys.argv:
        gpu_tests()
    output = ROOT/'manifests'/('phase04_cpu_tests.json' if '--cpu-only' in sys.argv else 'phase04_tests.json')
    output.write_text(json.dumps(dict(status='PASS', tests=rows, synthetic=synthetic), indent=2), encoding='utf-8')
    print('PHASE04_TESTS_PASS', flush=True)
