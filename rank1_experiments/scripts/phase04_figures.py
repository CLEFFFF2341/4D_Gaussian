"""Scientific diagnostics generated from actual test/run JSON, no edited scores."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]
out=ROOT/'diagnostics/phase04';out.mkdir(parents=True,exist_ok=True)
probe=json.loads((ROOT/'manifests/phase04_real_probe.json').read_text())
summary=json.loads((ROOT/'manifests/phase04_short_summary.json').read_text())
tests=json.loads((ROOT/'manifests/phase04_tests.json').read_text())
fig,axes=plt.subplots(2,2,figsize=(13,8),layout='constrained')
samples=[(s,c) for s in ('coffee_martini','cut_roasted_beef') for c in (('cam02','cam06','cam10','cam16') if s=='coffee_martini' else ('cam02','cam06','cam10','cam15'))]
x=np.arange(len(samples))
for j,k in enumerate((8,32,64)):
    values=[next(r['topk_mass_retention'] for r in probe['real_batch_results'] if r.get('k')==k and r['method']=='M' and r['scene']==s and r['camera']==c) for s,c in samples]
    axes[0,0].bar(x+(j-1)*.24,values,.24,label=f'K={k}')
axes[0,0].axhline(.95,color='black',ls='--',lw=1,label='camera mass gate')
axes[0,0].set(ylim=(.5,1.02),ylabel='Retained matched source alpha*T',title='K truncation: same initialized geometry')
axes[0,0].set_xticks(x,[f'{"C" if s=="coffee_martini" else "B"}/{c}' for s,c in samples],rotation=30)
axes[0,0].legend(fontsize=8)
cases=tests['synthetic'];x=np.arange(len(cases))
for j,method in enumerate(('B2','H','M')):
    axes[0,1].bar(x+(j-1)*.24,[r['losses'][method] for r in cases],.24,label=method)
axes[0,1].set_xticks(x,['static','opposite\nforeground','single\nsurface','transparent\nmean','reflection\nproxy'])
axes[0,1].set(ylabel='NLL with identical constants',title='Mechanism examples and counterexamples')
axes[0,1].legend()
methods=('F0','B1','B2','H','M');x=np.arange(len(methods))
groups=[(s,t) for s in ('coffee_martini','cut_roasted_beef') for t in ('frozen','native')]
for j,(scene,topology) in enumerate(groups):
    selected=[next(r for r in summary['runs'] if r['scene']==scene and r['topology']==topology and r['method']==m) for m in methods]
    label=f'{"coffee" if scene=="coffee_martini" else "beef"}/{topology}'
    axes[1,0].bar(x+(j-1.5)*.19,[r['train_and_save_seconds']/3 for r in selected],.19,label=label)
    axes[1,1].bar(x+(j-1.5)*.19,[r['peak_allocated_gib'] for r in selected],.19,label=label)
axes[1,0].set(ylabel='Seconds per complete updated frame',title='Prototype cost: training + save + diagnostics')
axes[1,1].set(ylabel='Peak PyTorch allocated GiB',title='RTX 4090 / 676 x 507 / K64')
for ax in axes[1]:ax.set_xticks(x,methods);ax.legend(fontsize=8)
for ax in axes.flat:ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
fig.suptitle('Phase04 correctness and short training; no quality improvement claim',fontsize=13)
fig.savefig(out/'implementation_diagnostics.png',dpi=150);plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
for ax,scene in zip(axes,('coffee_martini','cut_roasted_beef')):
    for method in ('B1','B2','H','M'):
        run=ROOT/'runs/phase04/native'/scene/method
        rows=[json.loads(line) for line in (run/'flow_diagnostics.jsonl').read_text().splitlines()]
        frames=[2,3,4]
        uid=[min(r['uid_mass_retention'] for r in rows if r['frame']==f) for f in frames]
        k=[min(r['topk_mass_retention'] for r in rows if r['frame']==f) for f in frames]
        ax.plot(frames,uid,marker='o',label=method+' UID')
        ax.plot(frames,k,ls='--',marker='.',label=method+' K64')
    ax.set(title=scene,ylim=(.99,1.0002),xticks=[2,3,4],xlabel='Complete frame boundary',ylabel='Minimum reported retained mass')
    ax.grid(alpha=.2);ax.legend(fontsize=7,ncol=2)
fig.suptitle('Native density changes: match identities first; full source transmittance')
fig.savefig(out/'native_uid_coverage.png',dpi=150);plt.close(fig)
print(str(out))
