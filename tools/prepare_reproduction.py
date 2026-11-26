#!/usr/bin/env python3
"""Prepare the pinned runtime sources and public inputs for MemAnchorWorld."""

import argparse
import hashlib
import shutil
from pathlib import Path, PurePosixPath
from tempfile import NamedTemporaryFile, TemporaryDirectory
from urllib.parse import quote
from urllib.request import Request, urlopen
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parents[1]
RUNTIME_REVISION = '14f5fe3bdd3428002d913988fbfd9b653a1a207e'
RUNTIME_ARCHIVE = (
    'https://codeload.github.com/MSunDYY/SparseWorld/zip/' + RUNTIME_REVISION
)
INPUT_REVISION = 'cb2a44a852adf1653bfe108a5f8d761ef962e97e'
INPUT_URL = (
    'https://huggingface.co/MSunDYY2001/SparseWorld/resolve/' + INPUT_REVISION
)
SHARED_INPUTS = {
    'ckpts/epoch_56.pth':
        '376df7a785b76d72f2aefb9c75fea8e1120dfebb49f3b7db2eaad84dd36c3015',
    'admlp/stp3_val/data_nuscene.pkl':
        '287c3fe4ddc0928b4fdd1174b348b4c67c4316826eeb16c549f43d7fd4533f0d',
}
SPLIT_INPUTS = {
    'train': {
        'data/nuscenes/bevdetv2-nuscenes_infos_train.pkl':
            '5156d223b2bd98815e1ccbd83974e6d05fff2c914e7697a117766a64c86487ee',
        'admlp/fengze_nuscenes_infos_train.pkl':
            '84426e190f4c1f397653aa0f41d79807dfc807bc24ee6d0b03ac7bf2cf5c5e',
        'occworld/nuscenes_infos_train_temporal_v3_scene.pkl':
            '119f1cebc6362d5f26a6974b36f1307eb34105ff899b1d49ee2bf0febb47c1f5',
    },
    'val': {
        'data/nuscenes/bevdetv2-nuscenes_infos_val.pkl':
            '1e16e8b1532448e35dbc187de080c52a9766181daa3b7aa6f07f62f3af69364e',
        'admlp/fengze_nuscenes_infos_val.pkl':
            '2afa1c1dbaad329b16203968ecff3923327658ef55c19b0ee562164799aa6065',
        'occworld/nuscenes_infos_val_temporal_v3_scene.pkl':
            'd174b9563adcfc2b0450d164dce108b11742197208436f70781d4187251da566',
    },
}


def open_url(url):
    return urlopen(Request(url, headers={'User-Agent': 'MemAnchorWorld'}), timeout=60)


def prepare_runtime():
    """Install shared source files without replacing repository files."""
    print('Preparing runtime sources...', flush=True)
    with TemporaryDirectory(prefix='memanchorworld-runtime-') as directory:
        archive = Path(directory) / 'source.zip'
        with open_url(RUNTIME_ARCHIVE) as response, archive.open('wb') as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
        with ZipFile(archive) as source:
            for item in source.infolist():
                _, _, name = item.filename.partition('/')
                if item.is_dir() or not (
                    name.startswith('mmdet3d/') or name == 'setup.py'
                ):
                    continue
                parts = PurePosixPath(name).parts
                if any(part in {'.', '..'} or '\\' in part or ':' in part for part in parts):
                    raise ValueError(f'Invalid archive path: {name}')
                target = ROOT.joinpath(*parts)
                if not target.resolve().is_relative_to(ROOT):
                    raise ValueError(f'Archive path leaves the repository: {name}')
                target.parent.mkdir(parents=True, exist_ok=True)
                try:
                    output = target.open('xb')
                except FileExistsError:
                    continue
                try:
                    with output, source.open(item) as entry:
                        shutil.copyfileobj(entry, output)
                except BaseException:
                    target.unlink()
                    raise
    print('Runtime sources are ready.')


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def prepare_inputs(split):
    inputs = dict(SHARED_INPUTS)
    for name in SPLIT_INPUTS if split == 'both' else (split,):
        inputs.update(SPLIT_INPUTS[name])
    for name, digest in inputs.items():
        target = ROOT / name
        if target.exists():
            if sha256(target) != digest:
                raise ValueError(f'Existing file does not match the pinned input: {name}')
            print(f'Already present: {name}', flush=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        print(f'Downloading {name}', flush=True)
        with NamedTemporaryFile(dir=target.parent, suffix='.part', delete=False) as output:
            temporary = Path(output.name)
            try:
                with output, open_url(f'{INPUT_URL}/{quote(name)}') as response:
                    shutil.copyfileobj(response, output, length=1024 * 1024)
                if sha256(temporary) != digest:
                    raise ValueError(f'Checksum mismatch: {name}')
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('runtime', help='prepare the source files needed for the build')
    inputs = commands.add_parser('inputs', help='download published metadata and initialization weights')
    inputs.add_argument('--split', choices=tuple(SPLIT_INPUTS) + ('both',), default='both')
    args = parser.parse_args()
    if args.command == 'runtime':
        prepare_runtime()
    else:
        prepare_inputs(args.split)


if __name__ == '__main__':
    main()
