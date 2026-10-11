"""Scene-paired metrics from complete trials, including worst frame examples."""
import json
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from phase05.common import read,write
screen=read(ROOT/'manifests/phase05_screen.json');w=screen['winners']
rows=[read(p) for p in (ROOT/'manifests/phase05_trials').glob('*.json')]
out=ROOT/'diagnostics/phase05';out.mkdir(exist_ok=True)
fig,axes=plt.subplots(2,2,figsize=(12,7),layout='constrained');examples=[]
for col,scene in enumerate(('coffee_martini','cut_roasted_beef')):
    selected={m:next(r for r in rows if r['scene']==scene and r['method']==m and r['ratio']==w[m]['ratio']) for m in w}
    frames={m:read(ROOT/r['run']/'evaluation/validation/per_frame.json') for m,r in selected.items()}
    for control in ('B2','H'):
        delta=np.array([m['metrics']['PSNR']-c['metrics']['PSNR'] for m,c in zip(frames['M'],frames[control])])
        axes[0,col].plot(range(1,91),delta,label='M - '+control)
        worst=int(delta.argmin());examples.append(dict(scene=scene,control=control,frame=worst+1,difference=float(delta[worst]),
                                                     label='least gain or decline; no frame treated as independent sample',seed=0))
    axes[0,col].axhline(0,color='black',lw=1);axes[0,col].legend();axes[0,col].set(title=scene,xlabel='Frame',ylabel='Paired validation PSNR difference (dB)')
    x=np.arange(4)
    axes[1,col].bar(x,[selected[m]['metrics']['PSNR'] for m in ('B1','B2','H','M')])
    axes[1,col].set_xticks(x,['B1','B2','H','M']);axes[1,col].set(ylabel='Mean whole-frame PSNR (dB)',title='Selected by same two-scene macro rule')
for ax in axes.flat:ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
fig.suptitle('Phase05 seed0 screening: repeated-frame curves, not independent samples')
fig.savefig(out/'seed0_paired.png',dpi=150);plt.close(fig)
write(ROOT/'manifests/phase05_least_gain_examples.json',examples)
print(str(out/'seed0_paired.png'))
