"""Predeclared scene-level selection. Adjacent frames are not independent samples."""
def select(rows,methods=('B1','B2','H','M')):
    winners={}
    for method in methods:
        options=[]
        for ratio in (.03,.1,.3):
            group=[r for r in rows if r['method']==method and r['ratio']==ratio]
            assert len(group)==2
            options.append(dict(method=method,ratio=ratio,
                 PSNR=sum(r['metrics']['PSNR'] for r in group)/2,
                 LPIPS=sum(r['metrics']['LPIPS'] for r in group)/2))
        best=max(o['PSNR'] for o in options)
        tied=[o for o in options if best-o['PSNR']<=1e-6]
        winners[method]=min(tied,key=lambda o:(o['LPIPS'],o['ratio']))
    return winners

def screen(rows,winners):
    differences=[]
    for scene in ('coffee_martini','cut_roasted_beef'):
        selected={m:next(r for r in rows if r['scene']==scene and r['method']==m and r['ratio']==winners[m]['ratio']) for m in winners}
        strong=max(('B1','B2','H'),key=lambda m:selected[m]['metrics']['PSNR'])
        m=selected['M'];c=selected[strong]
        dif=dict(scene=scene,strongest_control=strong,whole=m['metrics']['PSNR']-c['metrics']['PSNR'],
                 lpips=m['metrics']['LPIPS']-c['metrics']['LPIPS'],
                 boundary=None if m['regions']['boundary']['PSNR'] is None or c['regions']['boundary']['PSNR'] is None else m['regions']['boundary']['PSNR']-c['regions']['boundary']['PSNR'],
                 vs_B2=m['metrics']['PSNR']-selected['B2']['metrics']['PSNR'],vs_H=m['metrics']['PSNR']-selected['H']['metrics']['PSNR'])
        differences.append(dif)
    whole=max(d['whole'] for d in differences)>=.05 and min(d['whole'] for d in differences)>=-.05
    regional=all(d['boundary'] is not None and d['boundary']>=.10 and d['whole']>=-.05 for d in differences)
    lpips=max(d['lpips'] for d in differences)<=.002
    return dict(signal=bool((whole or regional) and lpips),whole_rule=whole,regional_rule=regional,lpips_rule=lpips,
                differences=differences,final_repeat_gate_passed=False)
