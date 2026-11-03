"""Inspect model release state or download a published, checksum-verified artifact.
Only HTTPS public sources in the manifest may be downloaded; external dependencies
are intentionally not downloaded automatically.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen

ROOT=Path(__file__).resolve().parents[1]
MANIFEST=ROOT/'models'/'manifest.json'

def sha256_file(path: Path) -> str:
    digest=hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()

def verify_file(path: Path, expected_bytes: int, expected_sha256: str) -> None:
    if path.stat().st_size != expected_bytes:
        raise ValueError(f'Size mismatch for {path}')
    got=sha256_file(path)
    if got.lower()!=expected_sha256.lower():
        raise ValueError(f'SHA-256 mismatch for {path}: {got}')

def download_artifact(model: dict, dest_root: Path) -> Path:
    if model['status']!='published':
        raise ValueError(f"{model['id']}: {model['status']}. No download is available.")
    name=model.get('file')
    if not isinstance(name,str) or Path(name).name!=name or name in ('.','..'):
        raise ValueError('Unsafe model filename')
    urls=model.get('sources',[])
    if not urls or not isinstance(urls[0],str) or urlparse(urls[0]).scheme!='https':
        raise ValueError('Only HTTPS model URLs can be downloaded')
    path=dest_root/name
    dest_root.mkdir(parents=True,exist_ok=True)
    if path.is_file():
        verify_file(path,model['size_bytes'],model['sha256'])
        return path
    temp=path.with_suffix(path.suffix+'.download')
    try:
        with urlopen(urls[0],timeout=120) as src, temp.open('wb') as dest:
            while chunk:=src.read(1024*1024):dest.write(chunk)
        verify_file(temp,model['size_bytes'],model['sha256'])
        temp.replace(path)
        return path
    finally:
        temp.unlink(missing_ok=True)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--id',help='Model id from manifest; omit to list all artifacts')
    ap.add_argument('--dest',type=Path,default=ROOT/'models'/'files')
    args=ap.parse_args()
    data=json.loads(MANIFEST.read_text(encoding='utf8'))
    for model in data['models']:
        print(f"{model['id']:26s}  {model['status']}")
    if args.id:
        match=[m for m in data['models'] if m['id']==args.id]
        if not match:ap.error('Unknown model id')
        print(download_artifact(match[0],args.dest))
if __name__=='__main__':main()
