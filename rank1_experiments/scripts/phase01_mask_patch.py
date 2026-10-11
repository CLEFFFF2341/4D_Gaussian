"""Prepare one isolated CUDA mask-stride correction and a reviewable diff."""
import difflib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
REL=Path('cuda_rasterizer/backward.cu')
ORIGINAL=ROOT/'sources/MoRGS-5c62ae88b8fb2e3d4213329dd1ec3a61c66f8902/submodules/gaussian-rasterization-grad'/REL
TARGET=ROOT/'build_sources/gaussian-rasterization-grad'/REL
old=ORIGINAL.read_text(encoding='utf-8')
assert old.count('update_mask[7 * idx')==10
assert old.count('update_mask[global_id * 7')==12
new=old.replace('update_mask[7 * idx','update_mask[8 * idx').replace('update_mask[global_id * 7','update_mask[global_id * 8')
TARGET.write_text(new,encoding='utf-8')
(ROOT/'patches').mkdir(exist_ok=True)
patch=''.join(difflib.unified_diff(old.splitlines(keepends=True),new.splitlines(keepends=True),
                 fromfile='a/submodules/gaussian-rasterization-grad/'+REL.as_posix(),
                 tofile='b/submodules/gaussian-rasterization-grad/'+REL.as_posix()))
(ROOT/'patches/mask_stride_8.patch').write_text(patch,encoding='utf-8')
print('Changed 22 mask-stride accesses in the isolated build source.')
