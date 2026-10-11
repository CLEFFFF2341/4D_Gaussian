import concurrent.futures, hashlib, json, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SHA = '5c62ae88b8fb2e3d4213329dd1ec3a61c66f8902'
REPO = 'LeeWonJoon9868/MoRGS'
FILES = ['README.md', 'train.py', 'configs/dynerf.yaml', 'arguments/__init__.py', 'gaussian_renderer/__init__.py', 'scene/gaussian_model.py', 'scene/utils.py', 'utils/loader_utils.py', 'scene/dataset_readers.py', 'metrics_video.py', '.gitmodules']

def fetch(path):
    url = f'https://raw.githubusercontent.com/{REPO}/{SHA}/{path}'
    try:
        request = urllib.request.Request(url, headers={'User-Agent': 'research-source-audit'})
        with urllib.request.urlopen(request, timeout=35) as r:
            data = r.read()
        target = ROOT / 'source_audit' / 'MoRGS' / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        return {'path': path, 'url': url, 'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data), 'status': 'retrieved'}
    except Exception as e:
        return {'path': path, 'url': url, 'status': 'failed', 'error': str(e)}

if __name__ == '__main__':
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        records = list(pool.map(fetch, FILES))
    request = urllib.request.Request(f'https://api.github.com/repos/{REPO}/git/trees/{SHA}?recursive=1', headers={'User-Agent': 'research-source-audit'})
    try:
        with urllib.request.urlopen(request, timeout=35) as response:
            tree = json.load(response)
        (ROOT/'source_audit'/'tree.json').write_text(json.dumps(tree, indent=2), encoding='utf-8')
    except Exception as e:
        records.append({'path': 'tree.json', 'status': 'failed', 'error': str(e)})
    (ROOT/'source_audit'/'acquisition.json').write_text(json.dumps({'repository': REPO, 'commit': SHA, 'records': records}, indent=2), encoding='utf-8')
    print(json.dumps([{'path':r['path'], 'status':r['status']} for r in records]))
