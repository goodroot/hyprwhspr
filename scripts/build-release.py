#!/usr/bin/env python3
"""Build and validate the application-only release, without importing runtime deps."""
import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'lib/src'))
from managed_install import VERSION, digest, verify_payload


def build(version, output):
    if not VERSION.fullmatch(version):
        raise ValueError('A stable semver tag is required')
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temporary:
        stage = Path(temporary) / 'payload'
        stage.mkdir()
        for name in ('bin', 'lib', 'config', 'share', 'scripts'):
            shutil.copytree(ROOT / name, stage / name, symlinks=True,
                            ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.pytest_cache'))
        for source in [*ROOT.glob('requirements*.txt'), *ROOT.glob('LICENSE*')]:
            if source.name != 'requirements-test.txt':
                shutil.copy2(source, stage / source.name)
        spec = importlib.util.spec_from_file_location('validator', ROOT / 'scripts/validate-package-payload.py')
        validator = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(validator)
        validator.validate(stage)
        files = {str(p.relative_to(stage)): digest(p) for p in stage.rglob('*') if p.is_file()}
        (stage / 'release.json').write_text(json.dumps({'format': 1, 'version': version, 'files': files}, indent=2))
        verify_payload(stage)
        archive = output / f'hyprwhspr-{version}.tar.gz'
        with tarfile.open(archive, 'w:gz') as bundle:
            for path in sorted(stage.rglob('*')):
                if path.is_symlink():
                    raise ValueError(f'Release symlinks are forbidden: {path}')
                bundle.add(path, arcname=str(path.relative_to(stage)), recursive=False)
        metadata = {'format': 1, 'version': version, 'archive': archive.name, 'sha256': digest(archive)}
        (output / 'hyprwhspr-release.json').write_text(json.dumps(metadata, indent=2) + '\n')
        (output / 'SHA256SUMS').write_text(f"{metadata['sha256']}  {archive.name}\n")
        # Recovery entry point is stdlib-only and published alongside the archive.
        shutil.copy2(ROOT / 'lib/src/managed_install.py', output / 'managed_install.py')
        with (output / 'SHA256SUMS').open('a') as stream:
            stream.write(f"{digest(output / 'managed_install.py')}  managed_install.py\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('version')
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    build(args.version, args.output)
