import torch
from scene.gaussian_model import GaussianModel
class TrackedModel(GaussianModel):

    def initialize_ids(self):
        self.uid = torch.arange(len(self.get_xyz), device='cuda', dtype=torch.long)
        self.ancestor_uid = self.uid.clone()
        self.origin = self.uid.clone()
        self.uid_clock = len(self.uid)
        self.current_step = 0
        self.topology_events = []

    def indices(self, ids):
        pos = torch.searchsorted(self.uid, ids)
        assert torch.equal(self.uid[pos], ids)
        return pos

    def reserve(self, n):
        ids = torch.arange(self.uid_clock, self.uid_clock + n, device='cuda', dtype=torch.long)
        self.uid_clock += n
        return ids

    def densification_postfix(self, *args, clone_mask=None, split_mask=None, N=2):
        source = self.uid[clone_mask] if clone_mask is not None else self.uid[split_mask].repeat(N)
        origins = self.origin[clone_mask] if clone_mask is not None else self.origin[split_mask].repeat(N)
        super().densification_postfix(*args, clone_mask=clone_mask, split_mask=split_mask, N=N)
        self.uid = torch.cat((self.uid, self.reserve(len(source))))
        self.ancestor_uid = torch.cat((self.ancestor_uid, source))
        self.origin = torch.cat((self.origin, origins))

    def prune_points(self, mask):
        super().prune_points(mask)
        keep = ~mask
        (self.uid, self.ancestor_uid, self.origin) = (self.uid[keep], self.ancestor_uid[keep], self.origin[keep])

    def assert_identity(self):
        groups = {g['name']: g['params'] for g in self.optimizer.param_groups}
        for (name, param) in self._latents.items():
            assert groups[name][0] is param, name
        assert self.offset_3d is None
        assert len(self.uid) == len(self.get_xyz) == len(self.origin)
        assert bool((self.uid[1:] > self.uid[:-1]).all())

    def accumulate(self, out):
        n = len(self.uid)
        (grad, radii) = (out['viewspace_points'].grad, out['radii'])
        visible = radii > 0
        self.max_radii2D[visible] = torch.maximum(self.max_radii2D[visible], radii[visible])
        self.xyz_gradient_accum[visible] += grad[visible, :2].norm(dim=-1, keepdim=True)
        self.denom[visible] += 1
