"""Verify active implementation and extension ownership after baseline cleanup."""
import ast,hashlib,json,re,sys,os
from pathlib import Path
import common,engine
BASE=Path(__file__).resolve().parent
ROOT=BASE.parent
suffixes={'.py','.cu','.cuh','.cpp','.h','.cmd','.ps1'}
forbidden=re.compile(r'\bn0[12]\b|morgs_n0[12]|n0[12]_rasterization|BranchState|MaterialChart|relative_d',re.I)
excluded={'envs','node_modules','.git','build','__pycache__','third_party','sources','tools','cache','data','data_recovery','downloads','runs','smoke','weights','wheels','wheels_cu121'}
guard_files={BASE/'smoke_test.py',Path(__file__).resolve()}
matches=[];paths=[];parsed=0
for directory,dirs,files in os.walk(ROOT,followlinks=False):
    dirs[:]=[name for name in dirs if name not in excluded]
    for name in files:
        path=Path(directory)/name
        if path.suffix not in suffixes and not path.name.endswith(('.py.disabled','.cmd.disabled')):continue
        relative=path.relative_to(ROOT)
        if path in guard_files:continue # Negative assertions are not implementations.
        text=path.read_text(encoding='utf-8-sig',errors='replace')
        if forbidden.search(text) or forbidden.search(path.name):matches.append(str(relative))
        if path.suffix=='.py':ast.parse(text);parsed+=1
        paths.append(path)
assert not matches,matches
assert not any(ROOT.glob('morgs_n0*'))
assert not any((ROOT/'rank1_experiments/archives').glob('n01*'))
extension=Path(engine.native._C.__file__).resolve()
assert extension.is_relative_to(BASE/'native/clean_rasterization')
exports=[name for name in dir(engine.native._C) if not name.startswith('_')]
assert sorted(exports)==['mark_visible','rasterize_gaussians','rasterize_gaussians_backward'],exports
assert not hasattr(engine.native,'CAPTURE')
for scene in ('coffee_martini','cut_roasted_beef'):
    cfg,model,quant,opt=common.configuration(scene,0)
    assert quant.param_names==['xyz','f_dc','f_rest','sc','rot','op']
    assert quant.quant_type[0]=='none'
    assert all(not getattr(model,key) for key in ('morgs_flow_supervision','morgs_motion_offset','morgs_motion_confidence'))
    assert not any('sam2' in key or 'flow_model' in key for key in cfg['model_params'])
    assert Path(cfg['source_config']).is_relative_to(BASE/'configs')
hashes={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths if path.is_relative_to(BASE) or path.name=='train.py'}
result=dict(passed=True,scanned_code_files=len(paths),parsed_python_files=parsed,
            residual_implementation_matches=matches,extension=str(extension),exports=exports,
            six_attributes=True,three_paper_modules_disabled=True,external_algorithm_dependencies=False,
            source_sha256=hashes,retired_code_outside_workspace='E:/4D_Gaussian_retired_code_20261012',
            physical_deletion_completed=False,deletion_review='blocked by policy')
(BASE/'source_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k!='source_sha256'},indent=2))
