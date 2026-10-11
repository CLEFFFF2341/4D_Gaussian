from pathlib import Path
from PIL import Image
import cv2
root=Path(__file__).resolve().parents[1]
for protocol in ['final','dev']:
    for p in sorted((root/f'data/dynerf/{protocol}/test/coffee_martini/cam00').glob('*.png')):
        try:
            with Image.open(p) as im: im.load()
        except Exception as e:
            x=cv2.imread(str(p)); print(str(p),repr(e),'opencv',None if x is None else x.shape,flush=True)
print('DONE',flush=True)
