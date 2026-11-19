#!/usr/bin/env python3
"""Read release contracts and coordinate checks for bump-version.sh.

preflight is read-only; wait may dispatch/rerun Actions, but never publishes a
release or changes a tag. All GitHub requests use the authenticated gh CLI.
"""
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
REPO = 'goodroot/hyprwhspr'
WORKFLOWS = (('release-application.yml', 'push'),
             ('python-dependency-smoke.yml', 'workflow_dispatch'))
FIELDS = 'databaseId,workflowDatabaseId,headSha,headBranch,event,status,conclusion,attempt,url'
DISCOVERY_TIMEOUT = 120
COMPLETION_TIMEOUT = 3 * 60 * 60
POLL_INTERVAL = 15


class ReleaseCheckError(RuntimeError):
    pass


def gh(*args, api=False):
    command = ['gh', *args]
    if not api:
        command.extend(['--repo', REPO])
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=60)
    except subprocess.CalledProcessError as exc:
        raise ReleaseCheckError(f"{' '.join(command)} failed: {exc.stderr.strip()}") from exc
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ReleaseCheckError(f"{' '.join(command)} failed: {exc}") from exc
    return result.stdout


def workflow_id(filename):
    data = json.loads(gh('api', f'repos/{REPO}/actions/workflows/{filename}', api=True))
    if data['state'] != 'active' or data['path'] != f'.github/workflows/{filename}':
        raise ReleaseCheckError(f'{filename} must be active on the default branch before bumping.')
    return data['id']


def expected_wheels():
    sys.path.insert(0, str(ROOT / 'lib/src'))
    import backend_installer as backend
    workflow = (ROOT / '.github/workflows/build-wheels.yml').read_text()
    versions = re.search(r"PYTHON_VERSIONS: '([^']+)'", workflow).group(1).split()
    expected = [f'3.{minor}' for minor in range(11, backend.MAX_COMPATIBLE_PYTHON[1] + 1)]
    if backend.MAX_COMPATIBLE_PYTHON[0] != 3 or versions != expected:
        raise ReleaseCheckError('Wheel Python matrix disagrees with the installer ceiling.')
    cuda = re.search(r"CUDA_BUILD_VERSION: '([^']+)'", workflow).group(1)
    variant = backend._get_wheel_variant(cuda)
    if not variant:
        raise ReleaseCheckError(f'No wheel variant for CUDA {cuda}')
    return [backend._get_wheel_filename(version, gpu, True)
            for version in versions for gpu in (variant, backend.VULKAN_WHEEL_VARIANT)]


def check_wheels():
    expected = expected_wheels()
    assets = json.loads(gh('release', 'view', 'wheels-v2', '--json', 'assets'))['assets']
    names = {asset['name'] for asset in assets}
    checksums = gh('release', 'download', 'wheels-v2', '--pattern', 'SHA256SUMS.txt', '--output', '-')
    sums = {}
    for number, line in enumerate(checksums.splitlines(), 1):
        if not line.strip():
            continue
        match = re.fullmatch(r'([a-fA-F0-9]{64}) [ *](\S+)', line)
        if not match:
            raise ReleaseCheckError(f'Malformed SHA256SUMS.txt entry on line {number}')
        digest, name = match.groups()
        if name in sums:
            raise ReleaseCheckError(f'Duplicate SHA256SUMS.txt entry: {name}')
        sums[name] = digest
    missing = []
    for name in expected:
        if name not in names:
            missing.append(f'{name}: asset missing')
        if name not in sums:
            missing.append(f'{name}: checksum missing')
    if missing:
        raise ReleaseCheckError('Wheels are not ready:\n  ' + '\n  '.join(missing) +
                                '\nFinish Build PyWhisperCPP Wheels, then rerun the bump script.')
    print(f'Wheel publication ready: {len(expected)} assets with checksums.', flush=True)


def matches(run, identifier, tag, commit, event):
    return (run['workflowDatabaseId'] == identifier and run['headBranch'] == tag
            and run['headSha'] == commit and run['event'] == event)


def ensure_run(filename, event, tag, commit):
    identifier = workflow_id(filename)
    deadline = time.monotonic() + DISCOVERY_TIMEOUT
    dispatched = False
    while True:
        runs = json.loads(gh('run', 'list', '--workflow', filename, '--branch', tag,
                             '--commit', commit, '--event', event, '--limit', '100', '--json', FIELDS))
        candidates = [run for run in runs if matches(run, identifier, tag, commit, event)]
        if candidates:
            run = max(candidates, key=lambda item: item['databaseId'])
            minimum_attempt = run['attempt']
            print(f"{filename}: {run['url']}", flush=True)
            if run['status'] == 'completed' and run['conclusion'] != 'success':
                if run['conclusion'] not in ('failure', 'cancelled', 'timed_out', 'startup_failure'):
                    raise ReleaseCheckError(f"{run['url']}: unexpected conclusion {run['conclusion']}")
                gh('run', 'rerun', str(run['databaseId']))
                minimum_attempt += 1
                print(f'Requested rerun, attempt {minimum_attempt}.', flush=True)
            return run, minimum_attempt, identifier, event
        if event == 'workflow_dispatch' and not dispatched:
            gh('workflow', 'run', filename, '--ref', tag)
            dispatched = True
        if time.monotonic() >= deadline:
            raise ReleaseCheckError(
                f'No matching {filename} run appeared for {tag} ({commit}). '
                f'Inspect https://github.com/{REPO}/actions/workflows/{filename}; '
                'rerun the bump script after resolving the trigger. Do not move the tag.')
        time.sleep(POLL_INTERVAL)


def wait_for_checks(tag, commit):
    pending = [ensure_run(filename, event, tag, commit) for filename, event in WORKFLOWS]
    deadline = time.monotonic() + COMPLETION_TIMEOUT
    while pending:
        remaining = []
        for original, minimum_attempt, identifier, event in pending:
            run = json.loads(gh('run', 'view', str(original['databaseId']), '--json', FIELDS))
            if run['databaseId'] != original['databaseId'] or not matches(run, identifier, tag, commit, event):
                raise ReleaseCheckError(f"Run identity changed: {original['url']}")
            # A rerun request may briefly still return the previous attempt.
            if run['attempt'] < minimum_attempt or run['status'] != 'completed':
                remaining.append((original, minimum_attempt, identifier, event))
            elif run['conclusion'] != 'success':
                raise ReleaseCheckError(
                    f"Validation {run['conclusion']}: {run['url']}. "
                    'Resolve the failure and rerun the bump script; the tag will not be moved.')
            else:
                print(f"Validation passed: {run['url']}", flush=True)
        pending = remaining
        if pending:
            if time.monotonic() >= deadline:
                raise ReleaseCheckError('Timed out waiting for checks: ' +
                                        ', '.join(item[0]['url'] for item in pending) +
                                        '. Runs remain active; rerun the bump script to resume.')
            time.sleep(POLL_INTERVAL)
    print(gh('release', 'view', tag, '--json', 'url', '--jq', '.url').strip(), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('preflight', 'wait'))
    parser.add_argument('--tag')
    parser.add_argument('--commit')
    args = parser.parse_args()
    try:
        if args.action == 'preflight':
            for filename, _ in WORKFLOWS:
                workflow_id(filename)
            check_wheels()
        else:
            if not args.tag or not re.fullmatch(r'v\d+\.\d+\.\d+', args.tag):
                parser.error('wait requires --tag vMAJOR.MINOR.PATCH')
            if not args.commit or not re.fullmatch(r'[a-f0-9]{40,64}', args.commit):
                parser.error('wait requires a full --commit SHA')
            wait_for_checks(args.tag, args.commit)
    except (ReleaseCheckError, ValueError, KeyError, TypeError, AttributeError) as exc:
        print(f'Release validation stopped: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
