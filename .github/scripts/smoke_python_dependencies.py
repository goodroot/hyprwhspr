"""Install and import one dependency plan in a disposable, isolated environment.

Run with --matrix to generate CI jobs from the runtime contract, or with a
plan key (including cli) to check it using the current Python. No models,
sidecars, desktop services or audio devices are used.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'lib/src'))

from backend_installer import MAX_COMPATIBLE_PYTHON
from dependency_plan import PLAN_SPECS, python_ceiling
from managed_install import CLI_IMPORTS, clean_env


def plans():
    return {'cli': ('requirements-cli.txt', CLI_IMPORTS, 'cli'), **PLAN_SPECS}


def matrix():
    return {'include': [{'plan': key, 'python': '.'.join(map(str,
                        python_ceiling(spec[2], MAX_COMPATIBLE_PYTHON)))}
                        for key, spec in plans().items()]}


def smoke(key):
    manifest, imports, family = plans()[key]
    extra = {'onnx': ('onnxruntime',), 'faster-whisper': ('ctranslate2',),
             'pywhispercpp': ('_pywhispercpp',)}.get(family, ())
    imports = tuple(dict.fromkeys((*imports, *extra)))
    env = clean_env()
    env.update(PIP_DISABLE_PIP_VERSION_CHECK='1', PIP_NO_INPUT='1', PIP_NO_CACHE_DIR='1',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    print(f'Plan: {key}; manifest: {manifest}; Python: {sys.version}', flush=True)
    with tempfile.TemporaryDirectory(prefix='hyprwhspr-dependency-smoke-') as temporary:
        environment = Path(temporary) / 'venv'
        subprocess.run([sys.executable, '-I', '-m', 'venv', str(environment)],
                       check=True, env=env)
        python = str(environment / 'bin/python')
        pip = [python, '-I', '-m', 'pip']
        try:
            subprocess.run([*pip, 'install', '-r', str(ROOT / manifest)], check=True, env=env)
            subprocess.run([*pip, 'check'], check=True, env=env)
            code = '\n'.join(f'import {name}\nprint({name!r}, "OK", flush=True)' for name in imports)
            subprocess.run([python, '-I', '-c', code], check=True, env=env)
        finally:
            # Retain the actual resolution even when installation or imports fail.
            subprocess.run([*pip, 'list', '--format=json'], check=False, env=env)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--matrix', action='store_true')
    parser.add_argument('plan', nargs='?', choices=tuple(plans()))
    args = parser.parse_args()
    if args.matrix:
        if args.plan:
            parser.error('--matrix cannot be combined with a plan')
        print(json.dumps(matrix()))
    elif args.plan:
        smoke(args.plan)
    else:
        parser.error('specify a plan or --matrix')


if __name__ == '__main__':
    main()
