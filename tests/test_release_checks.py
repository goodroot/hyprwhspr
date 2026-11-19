"""Release validation tests never contact GitHub or dispatch real workflows."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('release_checks', ROOT / 'scripts/release-checks.py')
checks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checks)
TAG = 'v1.2.3'
SHA = 'a' * 40


def run(identifier=1):
    return dict(databaseId=identifier * 10, workflowDatabaseId=identifier,
                headSha=SHA, headBranch=TAG, event='push' if identifier == 1 else 'workflow_dispatch',
                status='completed', conclusion='success', attempt=1,
                url=f'https://example.test/runs/{identifier}')


class ReleaseChecksTests(unittest.TestCase):
    def setUp(self):
        self.real_gh = checks.gh
        self.clock = 0
        self.calls = []
        self.runs = {1: [run(1)], 2: [run(2)]}
        self.sleep = mock.patch.object(checks.time, 'sleep', side_effect=self.advance)
        self.sleep.start()
        self.addCleanup(self.sleep.stop)
        timer = mock.patch.object(checks.time, 'monotonic', side_effect=lambda: self.clock)
        timer.start()
        self.addCleanup(timer.stop)
        gh = mock.patch.object(checks, 'gh', side_effect=self.github)
        gh.start()
        self.addCleanup(gh.stop)

    def advance(self, seconds):
        self.clock += seconds

    def github(self, *args, **kwargs):
        self.calls.append(args)
        if args[0] == 'api':
            filename = args[1].rsplit('/', 1)[-1]
            identifier = 1 if filename == 'release-application.yml' else 2
            return json.dumps({'id': identifier, 'state': 'active', 'path': f'.github/workflows/{filename}'})
        if args[:2] == ('run', 'list'):
            identifier = 1 if args[3] == 'release-application.yml' else 2
            return json.dumps(self.runs[identifier])
        if args[:2] == ('workflow', 'run'):
            self.runs[2] = [run(2)]
            return ''
        if args[:2] == ('run', 'rerun'):
            item = next(item for group in self.runs.values() for item in group if str(item['databaseId']) == args[2])
            item.update(attempt=item['attempt'] + 1, status='completed', conclusion='success')
            return ''
        if args[:2] == ('run', 'view'):
            return json.dumps(next(item for group in self.runs.values() for item in group if str(item['databaseId']) == args[2]))
        if args[:2] == ('release', 'view'):
            return 'https://example.test/release\n'
        self.fail(f'Unexpected gh call: {args}')

    def test_successful_runs_are_reused(self):
        checks.wait_for_checks(TAG, SHA)
        self.assertFalse(any(c[:2] in (('run', 'rerun'), ('workflow', 'run')) for c in self.calls))

    def test_missing_dependency_run_is_dispatched_at_tag(self):
        self.runs[2] = []
        checks.wait_for_checks(TAG, SHA)
        self.assertIn(('workflow', 'run', 'python-dependency-smoke.yml', '--ref', TAG), self.calls)

    def test_unrelated_runs_do_not_satisfy_validation(self):
        for field, value in [('headSha', 'b' * 40), ('headBranch', 'v9.9.9'),
                             ('event', 'pull_request'), ('workflowDatabaseId', 99)]:
            with self.subTest(field=field):
                self.runs[1] = [{**run(1), field: value}]
                with self.assertRaisesRegex(checks.ReleaseCheckError, 'No matching'):
                    checks.ensure_run('release-application.yml', 'push', TAG, SHA)

    def test_active_run_is_waited_without_dispatch(self):
        self.runs[1][0].update(status='in_progress', conclusion='')
        def finish(seconds):
            self.advance(seconds)
            self.runs[1][0].update(status='completed', conclusion='success')
        with mock.patch.object(checks.time, 'sleep', side_effect=finish):
            checks.wait_for_checks(TAG, SHA)
        self.assertFalse(any(c[:2] == ('workflow', 'run') for c in self.calls))

    def test_failed_and_cancelled_runs_are_rerun_once(self):
        for conclusion in ('failure', 'cancelled', 'timed_out'):
            with self.subTest(conclusion=conclusion):
                self.calls.clear()
                self.runs[1] = [{**run(1), 'conclusion': conclusion}]
                checks.wait_for_checks(TAG, SHA)
                self.assertEqual(self.calls.count(('run', 'rerun', '10')), 1)

    def test_rerun_does_not_accept_stale_previous_attempt(self):
        self.runs[1][0]['conclusion'] = 'failure'
        original = copy.deepcopy(self.runs[1][0])
        client = self.github
        def stale(*args, **kwargs):
            if args[:2] == ('run', 'view') and args[2] == '10' and self.clock == 0:
                return json.dumps(original)
            return client(*args, **kwargs)
        with mock.patch.object(checks, 'gh', side_effect=stale):
            checks.wait_for_checks(TAG, SHA)
        self.assertGreater(self.clock, 0)

    def test_new_failure_stops_without_automatic_retry_loop(self):
        self.runs[1][0].update(status='in_progress', conclusion='')
        def fail(seconds):
            self.advance(seconds)
            self.runs[1][0].update(status='completed', conclusion='failure')
        with mock.patch.object(checks.time, 'sleep', side_effect=fail):
            with self.assertRaisesRegex(checks.ReleaseCheckError, 'Validation failure'):
                checks.wait_for_checks(TAG, SHA)
        self.assertNotIn(('run', 'rerun', '10'), self.calls)

    def test_completion_timeout_leaves_runs_active(self):
        self.runs[1][0].update(status='queued', conclusion='')
        with mock.patch.object(checks, 'COMPLETION_TIMEOUT', 30):
            with self.assertRaisesRegex(checks.ReleaseCheckError, 'Timed out'):
                checks.wait_for_checks(TAG, SHA)
        self.assertEqual(self.runs[1][0]['status'], 'queued')

    def test_api_error_is_not_treated_as_missing_run(self):
        with mock.patch.object(checks, 'gh', side_effect=checks.ReleaseCheckError('API unavailable')):
            with self.assertRaisesRegex(checks.ReleaseCheckError, 'API unavailable'):
                checks.wait_for_checks(TAG, SHA)

    def test_delayed_application_discovery_and_latest_matching_run(self):
        self.runs[1] = []
        def appear(seconds):
            self.advance(seconds)
            self.runs[1] = [run(1), {**run(1), 'databaseId': 11}]
        with mock.patch.object(checks.time, 'sleep', side_effect=appear):
            found, _, _, _ = checks.ensure_run('release-application.yml', 'push', TAG, SHA)
        self.assertEqual(found['databaseId'], 11)

    def test_dependency_dispatch_timeout_does_not_dispatch_repeatedly(self):
        self.runs[2] = []
        client = self.github
        def never_appears(*args, **kwargs):
            if args[:2] == ('workflow', 'run'):
                self.calls.append(args)
                return ''
            return client(*args, **kwargs)
        with mock.patch.object(checks, 'gh', side_effect=never_appears):
            with self.assertRaisesRegex(checks.ReleaseCheckError, 'No matching'):
                checks.ensure_run('python-dependency-smoke.yml', 'workflow_dispatch', TAG, SHA)
        self.assertEqual(sum(c[:2] == ('workflow', 'run') for c in self.calls), 1)

    def test_viewed_run_must_keep_identity(self):
        client = self.github
        def wrong_commit(*args, **kwargs):
            if args[:2] == ('run', 'view'):
                return json.dumps({**run(1), 'headSha': 'b' * 40})
            return client(*args, **kwargs)
        with mock.patch.object(checks, 'gh', side_effect=wrong_commit):
            with self.assertRaisesRegex(checks.ReleaseCheckError, 'identity changed'):
                checks.wait_for_checks(TAG, SHA)

    def test_disabled_workflow_is_rejected_before_dispatch(self):
        with mock.patch.object(checks, 'gh', return_value=json.dumps(
                {'id': 1, 'state': 'disabled_manually', 'path': '.github/workflows/release-application.yml'})):
            with self.assertRaisesRegex(checks.ReleaseCheckError, 'must be active'):
                checks.workflow_id('release-application.yml')

    def test_gh_failures_are_actionable(self):
        with mock.patch.object(subprocess, 'run', side_effect=subprocess.CalledProcessError(1, ['gh'], stderr='denied')):
            with self.assertRaisesRegex(checks.ReleaseCheckError, 'denied'):
                self.real_gh('run', 'list')


class WheelReadinessTests(unittest.TestCase):
    def test_complete_assets_and_checksums_are_required(self):
        valid = 'a' * 64 + '  one.whl\n' + 'b' * 64 + '  two.whl\n'
        for names, sums, error in [(['one.whl', 'two.whl'], valid, None),
                                   (['one.whl'], valid, 'two.whl: asset missing'),
                                   (['one.whl', 'two.whl'], valid.splitlines()[0], 'two.whl: checksum missing'),
                                   (['one.whl', 'two.whl'], valid + valid, 'Duplicate'),
                                   (['one.whl', 'two.whl'], 'bad  one.whl', 'Malformed')]:
            with self.subTest(error=error), mock.patch.object(checks, 'expected_wheels', return_value=['one.whl', 'two.whl']), \
                    mock.patch.object(checks, 'gh', side_effect=[json.dumps({'assets': [{'name': n} for n in names]}), sums]):
                if error:
                    with self.assertRaisesRegex(checks.ReleaseCheckError, error):
                        checks.check_wheels()
                else:
                    checks.check_wheels()
