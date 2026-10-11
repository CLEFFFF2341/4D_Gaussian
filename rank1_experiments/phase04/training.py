"""Common training hook: fixed source state, current true RGB geometry, cached Ω."""
import csv
import json
import subprocess
from pathlib import Path
import torch
from .native import snapshot, raster_candidates, candidate_loss
from protocol_cache import TeacherCache

ROOT = Path(__file__).resolve().parents[1]


def motion_gradients(loss, pc, retain_graph=True):
    params = [pc._latents['xyz']]
    if pc.offset_3d is not None:
        params.append(pc.offset_3d)
    params += list(pc.latent_decoders['xyz'].parameters())
    # Offset must be the optimizer object, not just a differentiable side tensor.
    pc.assert_optimizer_identity()
    grads = torch.autograd.grad(loss, params, retain_graph=retain_graph, allow_unused=True)
    norms = [float(g.detach().norm()) if g is not None else 0. for g in grads]
    return dict(norm=sum(v*v for v in norms)**.5, xyz_norm=norms[0],
                offset_norm=norms[1] if pc.offset_3d is not None else None,
                decoder_norms=norms[2:] if pc.offset_3d is not None else norms[1:])


class SharedTraining:
    def __init__(self, args, pc):
        self.args, self.pc = args, pc
        pc.phase04_freeze_topology = args.flow_freeze_topology
        pc.ensure_identity()
        self.enabled = args.flow_method != 'F0'
        self.candidates, self.teachers, self.source = {}, {}, None
        self.logged = set()
        self.lambda_flow = (getattr(pc, 'phase04_flow_lambda', None) if args.flow_lambda < 0 else args.flow_lambda)
        scene = Path(args.source_path).name
        protocol = Path(args.source_path).parents[1].name
        split = json.loads((ROOT/'manifests'/f'splits_{protocol}.json').read_text())['scenes'][scene]
        self.cameras = dict(zip(split['flow_runtime_indices'], split['flow_cameras']))
        self.cache = TeacherCache(ROOT/'cache/teachers'/('dev_phase03' if protocol=='dev' else protocol),
                                  scene, split['flow_cameras'], 'shared', 90 if protocol=='dev' else 300)
        self.out = Path(args.model_path)
        self.out.mkdir(parents=True, exist_ok=True)
        resolved = vars(args).copy()
        if self.lambda_flow is not None:
            resolved['effective_flow_lambda'] = self.lambda_flow
            resolved['lambda_inherited_from_checkpoint'] = args.flow_lambda < 0
        resolved.update(shared_rgb_geometry='pc.get_xyz', source_detached=True, weights_detached=True,
                        out_sigma=128., projection='complete camera projection on source grid',
                        lambda_policy='fixed supplied' if self.lambda_flow is not None else 'first training batch gradient ratio; fixed thereafter',
                        gradient_parameters='xyz latent, actual optimized offset, xyz decoder',
                        initialization=args.start_checkpoint, baseline_artifact_lock='manifests/phase03_artifact_lock.json')
        (self.out/'resolved_config.json').write_text(json.dumps(resolved, indent=2), encoding='utf-8')
        import sys
        (self.out/'command.txt').write_text(subprocess.list2cmdline([sys.executable, *sys.argv]), encoding='utf-8')
        env = json.loads((ROOT/'manifests/environment.json').read_text())
        env['phase04_git_commit'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT).decode().strip()
        env['phase04_device'] = torch.cuda.get_device_name()
        (self.out/'environment.json').write_text(json.dumps(env, indent=2), encoding='utf-8')

    def begin_frame(self, frame):
        self.frame = frame
        self.candidates, self.teachers, self.logged = {}, {}, set()
        if frame > 1 and self.enabled:
            self.source = snapshot(self.pc, self.pc.ensure_identity().uid)

    def flow(self, camera, rgb_loss, iteration):
        if not self.enabled:
            # No call to raster_candidates, projection, loss, or grad-calibration.
            if camera.uid not in self.logged:
                self.write(dict(frame=self.frame, iteration=iteration, camera=camera.uid,
                                method='F0', flow_enabled=False, flow_loss=0., flow_grad_norm=0., lambda_flow=0.))
                self.logged.add(camera.uid)
            return rgb_loss.new_zeros(())
        version = self.pc.topology_version
        cached = self.candidates.get(camera.uid)
        candidate_changed = cached is None or cached[0] != version
        if candidate_changed:
            cached = (version, raster_candidates(self.source, camera, self.pc.identity.uid, self.args.flow_k))
            self.candidates[camera.uid] = cached
        if camera.uid not in self.teachers:
            self.teachers[camera.uid] = self.cache.pair(self.cameras[camera.uid], self.frame-2, self.frame-1, 'cuda')
        teacher = self.teachers[camera.uid]
        value, diagnostics = candidate_loss(self.source, self.pc.get_xyz, camera, cached[1],
                                             teacher['flow'], teacher['valid'], self.args.flow_method,
                                             self.args.flow_sigma, self.args.flow_epsilon)
        assert torch.isfinite(value), 'nonfinite shared flow loss'
        first = camera.uid not in self.logged
        if first or candidate_changed or self.lambda_flow is None:
            rgb_gradient = motion_gradients(rgb_loss, self.pc)
            flow_gradient = motion_gradients(value, self.pc)
            if self.lambda_flow is None:
                assert rgb_gradient['norm'] > 0 and flow_gradient['norm'] > 1e-12, (rgb_gradient, flow_gradient)
                self.lambda_flow = self.args.flow_grad_ratio*rgb_gradient['norm']/flow_gradient['norm']
                self.pc.phase04_flow_lambda = self.lambda_flow
                cfg_path = self.out/'resolved_config.json'
                cfg = json.loads(cfg_path.read_text())
                cfg.update(effective_flow_lambda=self.lambda_flow, calibration_frame=self.frame,
                           calibration_camera=camera.uid, calibration_rgb=rgb_gradient, calibration_flow=flow_gradient)
                cfg_path.write_text(json.dumps(cfg, indent=2), encoding='utf-8')
            self.write(dict(frame=self.frame, iteration=iteration, camera=camera.uid, **diagnostics,
                            flow_loss=float(value.detach()), rgb_loss=float(rgb_loss.detach()),
                            flow_grad_norm=flow_gradient['norm'], rgb_grad_norm=rgb_gradient['norm'],
                            flow_xyz_grad_norm=flow_gradient['xyz_norm'], flow_offset_grad_norm=flow_gradient['offset_norm'],
                            lambda_flow=self.lambda_flow, points=len(self.pc.identity.uid),
                            topology_version=version, freeze_topology=self.args.flow_freeze_topology))
            self.logged.add(camera.uid)
        return value*self.lambda_flow

    def write(self, row):
        with (self.out/'flow_diagnostics.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(row)+'\n')
        print('PHASE04_FLOW '+json.dumps(row), flush=True)

    def end_frame(self, frame):
        from phase03_hooks import atomic_save
        atomic_save(dict(uid=self.pc.identity.uid.detach().cpu(),
                         parent=self.pc.identity.parent.detach().cpu(),
                         next_uid=self.pc.identity.next_uid,
                         topology_version=self.pc.topology_version),
                    self.out/'identities'/f'{frame:04d}.pt')
        if self.enabled:
            # Release candidate graphs/storage before complete boundary checkpoint.
            self.candidates, self.teachers, self.source = {}, {}, None
        resources = [json.loads(line) for line in (self.out/'resources.jsonl').read_text().splitlines()]
        if resources:
            with (self.out/'metrics_per_frame.csv').open('w', newline='', encoding='utf-8') as handle:
                writer = csv.DictWriter(handle, fieldnames=list(resources[0]))
                writer.writeheader(); writer.writerows(resources)
            (self.out/'resource.json').write_text(json.dumps(dict(frames=resources,
                train_and_save_seconds=sum(r['train_and_save_seconds'] for r in resources),
                peak_allocated=max(r['peak_allocated'] for r in resources),
                peak_reserved=max(r['peak_reserved'] for r in resources)), indent=2), encoding='utf-8')
