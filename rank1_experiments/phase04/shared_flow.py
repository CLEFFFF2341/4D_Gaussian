"""Reference candidate composition, persistent identity, projection and shared losses."""
import math
from dataclasses import dataclass
import torch


@dataclass
class Identity:
    uid: torch.Tensor
    parent: torch.Tensor
    next_uid: int

    @classmethod
    def initial(cls, count, device='cpu'):
        return cls(torch.arange(count, device=device), torch.full((count,), -1, device=device), count)

    def append(self, parents):
        parents = parents.to(self.uid.device).long()
        new = torch.arange(self.next_uid, self.next_uid + len(parents), device=self.uid.device)
        self.next_uid += len(parents)
        self.uid = torch.cat((self.uid, new))
        self.parent = torch.cat((self.parent, parents))

    def select(self, indices):
        self.uid, self.parent = self.uid[indices], self.parent[indices]


def match_uid(source, target):
    """Return source -> current row; ancestry never supplies a match."""
    if not len(target):
        return torch.full_like(source, -1)
    ordered, indices = target.sort()
    pos = torch.searchsorted(ordered, source)
    safe = pos.clamp(max=len(target)-1)
    return torch.where((pos < len(target)) & (ordered[safe] == source), indices[safe], -1)


def project(xyz, full_projection, view, width, height):
    """MoRGS row-vector matrices and ndc2Pix: ((ndc+1)*size-1)/2."""
    homogeneous = torch.cat((xyz, torch.ones_like(xyz[:, :1])), -1)
    clip = homogeneous @ full_projection
    camera = homogeneous @ view
    denom = clip[:, 3:4]
    safe = torch.where(denom.abs() > 1e-7, denom, torch.full_like(denom, 1e-7))
    ndc = clip[:, :2] / (safe + 1e-7)  # pinned RGB preprocess uses w + 1e-7
    size = xyz.new_tensor([width, height])
    pixels = ((ndc + 1) * size - 1) * .5
    valid = (camera[:, 2] > .2) & (denom[:, 0] > 1e-7) & torch.isfinite(pixels).all(-1)
    return pixels, valid.detach()


@torch.no_grad()
def reference_candidates(xy, conic, point_list, ranges, height, width, alive, k=8, tile=16):
    """Small CPU oracle with the pinned renderer's thresholds and termination.

    T includes every accepted contribution, including unmatched/pruned identities.
    Select top K only among surviving one-to-one identities, after full composition.
    """
    ids = torch.full((height, width, k), -1, dtype=torch.long, device=xy.device)
    weights = xy.new_zeros((height, width, k))
    mass = xy.new_zeros((height, width, 2))
    for y in range(height):
        for x in range(width):
            start, end = ranges[(y//tile)*((width+tile-1)//tile)+x//tile].tolist()
            trans = 1.
            candidates = []
            for row in point_list[start:end].tolist():
                dx, dy = xy[row] - xy.new_tensor([x, y])
                a, b, c, opacity = conic[row]
                power = -.5*(a*dx*dx+c*dy*dy)-b*dx*dy
                if power > 0:
                    continue
                alpha = min(.99, float(opacity*power.exp()))
                if alpha < 1/255:
                    continue
                next_trans = trans*(1-alpha)
                if next_trans < .0001:
                    break
                weight = alpha*trans
                mass[y, x, 0] += weight
                if alive[row]:
                    mass[y, x, 1] += weight
                    candidates.append((weight, row))
                trans = next_trans
            candidates.sort(key=lambda item: -item[0])  # stable depth-order tie break
            for j, (weight, row) in enumerate(candidates[:k]):
                ids[y, x, j], weights[y, x, j] = row, weight
    return ids, weights, mass


def flow_loss(flows, weights, teacher, valid, method='M', sigma=2., epsilon=.02,
              out_sigma=128., scale=1., charbonnier=1e-6, reference_double=False):
    """flows [P,K,2], weights [P,K], teacher [P,2]; full fixed Ω denominator.

    For B1 abstention is scored with the same fixed outlier NLL as the NLL
    families. Its units differ and lambda must be gradient-calibrated. F0 does
    not enter candidate/loss calculation in the training adapter.
    """
    if method not in ('F0', 'B1', 'B2', 'H', 'M'):
        raise ValueError(method)
    if not (sigma > 0 and out_sigma > 0 and 0 < epsilon < 1 and scale > 0):
        raise ValueError('Invalid global likelihood constants')
    if method == 'F0':
        return flows.new_zeros(()), {'method': method, 'flow_enabled': False}
    dtype = torch.float64 if reference_double else torch.float32
    m, y = flows.to(dtype)/scale, teacher.detach().to(dtype)/scale
    w = weights.detach().to(dtype).clamp_min(0)
    w = torch.where(torch.isfinite(w), w, torch.zeros_like(w))
    total = w.sum(-1, keepdim=True)
    covered = total[:, 0] > 0
    a = w/total.clamp_min(torch.finfo(dtype).tiny)
    # Avoid 0 * inf and log(0) derivative singularities; invalid observations are
    # outside Ω. Squared residuals are safely bounded above FP32's useful range.
    m = torch.nan_to_num(m, nan=0., posinf=1e15, neginf=-1e15).clamp(-1e15, 1e15)
    y = torch.nan_to_num(y, nan=0., posinf=1e15, neginf=-1e15).clamp(-1e15, 1e15)
    log_out = math.log(epsilon)-math.log(2*math.pi*out_sigma**2)-y.square().sum(-1)/(2*out_sigma**2)
    log_prior = torch.where(w > 0, a.clamp_min(torch.finfo(dtype).tiny).log(), -torch.inf)
    log_component = (math.log(1-epsilon) + log_prior - math.log(2*math.pi*sigma**2)
                     -(m-y[:, None]).square().sum(-1)/(2*sigma**2))
    all_components = torch.cat((log_component, log_out[:, None]), -1)
    mixture_logp = torch.logsumexp(all_components, -1)
    mean = (a[..., None]*m).sum(1)
    if method == 'M':
        per_pixel = -mixture_logp
    elif method == 'H':
        per_pixel = -all_components.max(-1).values
    elif method == 'B2':
        log_mean = (math.log(1-epsilon)-math.log(2*math.pi*sigma**2)
                    -(mean-y).square().sum(-1)/(2*sigma**2))
        log_mean = torch.where(covered, log_mean, -torch.inf)
        per_pixel = -torch.logsumexp(torch.stack((log_mean, log_out), -1), -1)
    else:
        endpoint = ((mean-y).square().sum(-1)+charbonnier**2).sqrt()
        per_pixel = torch.where(covered, endpoint, -log_out)
    mask = valid.detach().bool() & torch.isfinite(teacher).all(-1)
    denom = mask.sum().clamp_min(1)
    loss = torch.where(mask, per_pixel, torch.zeros_like(per_pixel)).sum()/denom
    with torch.no_grad():
        posterior = (all_components-mixture_logp[:, None]).exp()
        diag = dict(method=method, valid_pixels=int(mask.sum()),
                    candidate_pixels=int((mask & covered).sum()),
                    abstain_pixels=int((mask & ~covered).sum()),
                    candidate_coverage=float((mask & covered).sum()/denom),
                    outlier_posterior=float(posterior[mask, -1].mean()) if mask.any() else 0.,
                    responsibility_entropy=float((-(posterior*posterior.clamp_min(1e-30).log()).sum(-1))[mask].mean()) if mask.any() else 0.)
    return loss, diag


def resolution_scale(width, height):
    return math.hypot(width, height)/1024


def resize_flow(flow, height, width):
    old_h, old_w = flow.shape[-2:]
    result = torch.nn.functional.interpolate(flow[None], size=(height, width), mode='bilinear', align_corners=False)[0]
    return result*flow.new_tensor([width/old_w, height/old_h])[:, None, None]
