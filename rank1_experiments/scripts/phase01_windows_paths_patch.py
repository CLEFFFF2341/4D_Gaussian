"""Preserve upstream camera names while accepting Windows separators."""
import ast
import difflib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'sources/MoRGS-5c62ae88b8fb2e3d4213329dd1ec3a61c66f8902'
WORK=ROOT/'working/MoRGS'
relative=Path('scene/cameras.py')
old=(SOURCE/relative).read_text(encoding='utf-8')
new=old
for function in ['imageName_from_Path','camName_from_Path']:
    needle=f'def {function}(path):\n'
    assert new.count(needle)==1
    new=new.replace(needle,needle+'    path = str(path).replace("\\\\", "/")\n')
(WORK/relative).write_text(new,encoding='utf-8')
patch=''.join(difflib.unified_diff(old.splitlines(keepends=True),new.splitlines(keepends=True),
                                fromfile='a/'+relative.as_posix(),tofile='b/'+relative.as_posix()))
(ROOT/'patches/windows_camera_paths.patch').write_text(patch,encoding='utf-8')
tree=ast.parse(new)
functions=[x for x in tree.body if isinstance(x,ast.FunctionDef)
           and x.name in ['imageName_from_Path','camName_from_Path']]
namespace={'re':__import__('re')}
exec(compile(ast.Module(body=functions,type_ignores=[]),str(relative),'exec'),namespace)
cases=[('E:/data/dynerf/coffee/cam02/0000.png','cam02','cam02_0000'),
       ('E:\\data\\dynerf\\coffee\\cam02\\0000.png','cam02','cam02_0000'),
       ('/data/meetroom/camera_03/images/000017.png','camera_03','camera_03_000017'),
       ('E:\\data\\meetroom\\camera_03\\images\\000017.png','camera_03','camera_03_000017')]
for path,camera,image in cases:
    assert namespace['camName_from_Path'](path)==camera
    assert namespace['imageName_from_Path'](path)==image
result=dict(status='PASS',cases=len(cases),scope='Separator normalization only; preserves camera and frame names.')
(ROOT/'manifests/windows_camera_paths_test.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result))
