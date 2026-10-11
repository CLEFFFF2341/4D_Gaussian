"""Shared UID and optimizer-identity correction, separate from likelihood choice."""
import torch
from scene.gaussian_model import GaussianModel
from .shared_flow import Identity


class SharedGaussianModel(GaussianModel):
    def ensure_identity(self):
        if not hasattr(self, 'identity'):
            self.identity = Identity.initial(len(self._latents['xyz']), self._latents['xyz'].device)
            self.topology_version = 0
        assert len(self.identity.uid) == len(self._latents['xyz'])
        return self.identity

    def prune_points(self, mask):
        self.ensure_identity()
        if getattr(self, 'phase04_freeze_topology', False) and self.frame_idx > 1:
            return
        super().prune_points(mask)
        # Original prune_points sliced the rendered tensor after replacing the
        # optimizer Parameter. Rebind to the actual replacement for every method.
        if self.offset_3d is not None:
            self.offset_3d = next(g['params'][0] for g in self.optimizer.param_groups if g['name'] == 'offset_3d')
        self.identity.select(~mask)
        if bool(mask.any()):
            self.topology_version += 1
        self.assert_optimizer_identity()

    def densification_postfix(self, *args, **kwargs):
        self.ensure_identity()
        clone, split = kwargs.get('clone_mask'), kwargs.get('split_mask')
        n = kwargs.get('N', 2)
        if clone is not None:
            parents = self.identity.uid[clone].clone()
        elif split is not None:
            parents = self.identity.uid[split].repeat(n).clone()
        else:
            parents = torch.full((len(args[0]),), -1, device=self.identity.uid.device)
        super().densification_postfix(*args, **kwargs)
        assert len(parents) == len(args[0])
        self.identity.append(parents)
        if len(parents):
            self.topology_version += 1
        self.assert_optimizer_identity()

    def densify_dynamic(self, *args, **kwargs):
        if getattr(self, 'phase04_freeze_topology', False):
            return
        return super().densify_dynamic(*args, **kwargs)

    def assert_optimizer_identity(self):
        for group in self.optimizer.param_groups:
            if group['name'] in self.param_names:
                assert group['params'][0] is self._latents[group['name']], group['name']
            if group['name'] == 'offset_3d':
                assert group['params'][0] is self.offset_3d, 'offset_3d disconnected'
