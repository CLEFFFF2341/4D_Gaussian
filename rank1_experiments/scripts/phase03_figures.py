"""Scientific QA figures from actual held-out renders and measured resources."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
output=ROOT/'figures/phase03'
output.mkdir(parents=True,exist_ok=True)
runs=[('dev','coffee_martini','seed0_loader_exact','validation','cam01',89),
      ('dev','cut_roasted_beef','seed0_loader_exact','validation','cam01',89),
      ('final','coffee_martini','seed0_dense','test','cam00',299)]
for protocol,scene,tag,role,camera,source_frame in runs:
    run=ROOT/'runs/B0'/protocol/scene/tag
    prediction=run/'evaluation'/role/'renders'/camera/f'{source_frame:04d}.png'
    if not prediction.exists():continue
    gt=np.asarray(Image.open(ROOT/'data/dynerf'/protocol/role/scene/camera/f'{source_frame:04d}.png').convert('RGB'))/255.
    pred=np.asarray(Image.open(prediction).convert('RGB'))/255.
    error=np.abs(gt-pred).mean(-1)
    fig,axes=plt.subplots(1,3,figsize=(15,4.3))
    axes[0].imshow(gt);axes[0].set_title('Held-out ground truth')
    axes[1].imshow(pred);axes[1].set_title('B0 compatibility render')
    im=axes[2].imshow(error,vmin=0,vmax=.15,cmap='magma');axes[2].set_title('Mean absolute RGB error (0 to .15)')
    for ax in axes:ax.axis('off')
    fig.colorbar(im,ax=axes[2],fraction=.04,pad=.02)
    fig.suptitle(f'{scene} | {protocol} | {camera} | source frame {source_frame}',fontsize=13)
    fig.tight_layout()
    fig.savefig(output/f'{protocol}_{scene}_heldout.png',dpi=150)
    plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(11,4.3))
for protocol,scene,tag,role,camera,source_frame in runs:
    path=ROOT/'runs/B0'/protocol/scene/tag/'resources.jsonl'
    if not path.exists():continue
    records=[json.loads(x) for x in path.read_text(encoding='utf-8').splitlines()]
    x=[r['frame'] for r in records]
    axes[0].plot(x,[r['peak_allocated']/2**30 for r in records],label=f'{protocol} {scene}')
    # Exclude initial10k from the update-time panel; it is separately reported.
    axes[1].plot(x[1:],[r['train_and_save_seconds'] for r in records[1:]],label=f'{protocol} {scene}')
axes[0].set(xlabel='Model frame',ylabel='Peak allocated CUDA GiB')
axes[1].set(xlabel='Model frame',ylabel='Update + save seconds')
for ax in axes:ax.grid(alpha=.2);ax.legend(fontsize=8)
fig.tight_layout();fig.savefig(output/'resource_curves.png',dpi=160);plt.close(fig)
fig,axes=plt.subplots(1,3,figsize=(14,4.2))
for protocol,scene,tag,role,camera,source_frame in runs:
    path=ROOT/'runs/B0'/protocol/scene/tag/'evaluation'/role/'per_frame.json'
    if not path.exists():continue
    records=json.loads(path.read_text(encoding='utf-8'))
    for ax,metric in zip(axes,['PSNR','SSIM','LPIPS']):
        ax.plot([r['source_frame'] for r in records],
                [r['metrics']['unified_fp32'][metric] for r in records],label=f'{protocol} {scene}')
        ax.set(xlabel='Source frame',ylabel=f'Unified {metric}');ax.grid(alpha=.2)
axes[0].legend(fontsize=8)
fig.tight_layout();fig.savefig(output/'quality_curves.png',dpi=160);plt.close(fig)
print(str(output))
