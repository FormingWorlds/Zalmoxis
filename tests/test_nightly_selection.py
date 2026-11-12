"""The per-file nightly jobs that select ``accept_solves`` name their files, so a marked test
elsewhere would run in no job; the ``accept_solves_long`` and ``accept_solves_newton_setups``
jobs select over the whole tree, the latter split by test id (``-k``) over several jobs. See
docs/How-to/build_tests.md for the test tiers."""

from __future__ import annotations

import argparse
import ast
import itertools
import re
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


WORKFLOW = ROOT / '.github' / 'workflows' / 'nightly.yml'


def _suites(workflow):
    """Matrix entry of each nightly suite, by suite name; the matrix ends at the first line
    indented by 4 spaces or less after ``include:``."""
    matrix = re.split(r'\n {0,4}\S', workflow[workflow.index('include:') :], maxsplit=1)[0]
    return {j.split('\n', 1)[0].strip(): j for j in re.split(r'\n\s*- suite:', matrix)[1:]}


def _job_limit(job):
    """The job_timeout [min] of one matrix entry (quotes and a trailing comment allowed)."""
    limit = re.search(r'^ +job_timeout:\s*["\']?(\d+)["\']?\s*(#.*)?$', job, re.M)
    assert limit, f'no job_timeout in {job.splitlines()[0].strip()}'
    return int(limit.group(1))


def _entry(job):
    """(select, test files named in the job) of one matrix entry."""
    select = re.search(r'select:(.*)', job)
    return select.group(1).strip() if select else '', re.findall(r'tests/([\w/]+\.py)', job)


def _keyword(job):
    """The ``-k`` expression that pytest reads from the args of one matrix entry (after the
    YAML comment and quotes; the last ``-k``, also in a flag group such as ``-xk``), or None."""
    args = re.search(r'^\s*args:(.*)$', job, re.M)
    words = shlex.split(args.group(1), comments=True) if args else []
    if len(words) == 1:
        words = shlex.split(words[0])
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-k')
    for flag in 'lqsvx':
        parser.add_argument(f'-{flag}', action='count')
    for flag in 'cmnopr':
        parser.add_argument(f'-{flag}')
    return parser.parse_known_args(words)[0].k


def _jobs(workflow):
    """(select, test files named in the job) of each nightly suite; the entries that split one
    select by ``-k`` count once (``_newton_faults`` checks that they split it)."""
    jobs = []
    for job in _suites(workflow).values():
        entry = _entry(job)
        if _keyword(job) is None or entry not in jobs:
            jobs.append(entry)
    return jobs


def _accept_select(workflow):
    """The one select of the jobs that name files and take accept_solves."""
    selects = {sel for sel, files in _jobs(workflow) if files and _takes_accept(sel)}
    assert len(selects) == 1, selects
    return selects.pop()


def _takes_accept(select):
    return bool(re.search(r'(?<!not )\baccept_solves\b', select))


def _wolfbower_covered(workflow):
    """Whether accept_solves tests with the wolfbower mark run: the accept jobs leave them out
    and a job selects wolfbower over the whole test tree."""
    whole = any(sel == 'wolfbower' and not files for sel, files in _jobs(workflow))
    return whole and 'not wolfbower' in _accept_select(workflow)


def _marked(tests_dir, select):
    """Files under ``tests_dir``, as paths below it, with a test that ``-m select`` collects.
    A file that skips at import (a module-level importorskip) is not collected."""
    done = subprocess.run(
        [sys.executable, '-m', 'pytest', '--collect-only', '-q', '-m', select]
        + ['-o', 'addopts=', '-p', 'no:cacheprovider', '--rootdir', str(tests_dir), '.'],
        cwd=tests_dir,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert done.returncode in (0, 5), done.stdout[-2000:] + done.stderr[-2000:]
    return {line.split('::')[0] for line in done.stdout.splitlines() if '::' in line}


def _unselected(marked, workflow):
    """Files of ``marked`` that no nightly job whose select takes accept_solves names."""
    named = {f for sel, files in _jobs(workflow) if _takes_accept(sel) for f in files}
    return sorted(marked - named)


@pytest.mark.smoke
def test_every_accept_solves_file_runs_in_a_nightly_job():
    """Collects the whole test tree (about 20 s), so it runs with the smoke tests."""
    workflow = WORKFLOW.read_text()
    marked = _marked(ROOT / 'tests', _accept_select(workflow))
    assert {'test_invariants.py', 'test_accepted_on.py'} <= marked
    assert _unselected(marked, workflow) == []


@pytest.mark.unit
def test_a_marked_file_outside_the_named_jobs_is_found(tmp_path):
    workflow = WORKFLOW.read_text()
    for name in ('test_invariants.py', 'test_accepted_on.py', 'test_third.py'):
        (tmp_path / name).write_text(
            'import pytest\n\n@pytest.mark.accept_solves\ndef test_x(): pass\n'
        )
    (tmp_path / 'test_plain.py').write_text('def test_y(): pass\n')
    assert _unselected(_marked(tmp_path, _accept_select(workflow)), workflow) == [
        'test_third.py'
    ]


_FORMS = {
    'list': 'import pytest\n\npytestmark = [\n    pytest.mark.unit,\n'
    '    pytest.mark.accept_solves,\n]\n\n\ndef test_x():\n    pass\n',
    'param': 'import pytest\n\n\n@pytest.mark.parametrize(\n'
    '    "a", [pytest.param(1, marks=pytest.mark.accept_solves)]\n)\n'
    'def test_x(a):\n    pass\n',
    'alias': 'from pytest import mark\n\n\n@mark.accept_solves\ndef test_x():\n    pass\n',
    'sub/': 'import pytest\n\n\n@pytest.mark.accept_solves\ndef test_x():\n    pass\n',
}


@pytest.mark.unit
@pytest.mark.parametrize('form', sorted(_FORMS))
def test_every_form_of_the_marker_is_found(tmp_path, form):
    """A pytestmark list, a marked pytest.param, an aliased mark and a file in a subfolder all
    put a test in the accept_solves selection, so each is found when no job names it."""
    workflow = WORKFLOW.read_text()
    name = 'sub/test_third.py' if form == 'sub/' else 'test_third.py'
    (tmp_path / name).parent.mkdir(exist_ok=True)
    (tmp_path / name).write_text(_FORMS[form])
    (tmp_path / 'test_plain.py').write_text('def test_y():\n    pass\n')
    assert _unselected(_marked(tmp_path, _accept_select(workflow)), workflow) == [name]


@pytest.mark.unit
def test_a_marked_wolfbower_test_runs_in_the_wolfbower_job(tmp_path):
    """A test with both marks is collected by accept_solves alone (the canary) but not by
    the select of the accept jobs, and the workflow has a whole-tree wolfbower job."""
    workflow = WORKFLOW.read_text()
    (tmp_path / 'test_third.py').write_text(
        'import pytest\n\n\n@pytest.mark.accept_solves\n@pytest.mark.wolfbower\n'
        'def test_x():\n    pass\n'
    )
    assert _marked(tmp_path, 'accept_solves') == {'test_third.py'}
    assert _unselected(_marked(tmp_path, _accept_select(workflow)), workflow) == []
    assert _wolfbower_covered(workflow)


@pytest.mark.unit
@pytest.mark.parametrize(
    'old, new',
    [
        pytest.param(
            'select: wolfbower\n', 'select: wolfbower and slow\n', id='no_wolfbower_job'
        ),
        pytest.param(
            'not accept_solves_newton_setups and not wolfbower',
            'not accept_solves_newton_setups',
            id='accept_takes_wolfbower',
        ),
    ],
)
def test_a_workflow_that_drops_the_wolfbower_split_is_caught(old, new):
    """Without the whole-tree wolfbower job, or with accept jobs that take wolfbower tests,
    the both-marked tests are not covered as this file assumes."""
    workflow = WORKFLOW.read_text()
    assert old in workflow
    assert not _wolfbower_covered(workflow.replace(old, new))


_MARKS = (
    'unit',
    'smoke',
    'integration',
    'slow',
    'wolfbower',
    'paleos_adiabat',
    'accept_solves',
    'accept_solves_long',
    'accept_solves_newton_setups',
)
_SPECIAL = set(_MARKS) - {'unit', 'smoke', 'integration', 'slow'}


def _partition_faults(workflow):
    """Marker sets, with each named file and an unnamed one, that no job or more than one job
    selects. A set counts when it has a special mark, or unit, smoke or integration without
    slow. In an unnamed file, a set that the per-file accept jobs take runs in no job; the
    collection tests above cover that case."""
    from _pytest.mark.expression import Expression

    jobs = [(Expression.compile(sel), files) for sel, files in _jobs(workflow)]
    accept = Expression.compile(_accept_select(workflow))
    named = sorted({f for _, files in jobs for f in files})
    faults = []
    for bits in itertools.product((False, True), repeat=len(_MARKS)):
        marks = {m for m, on in zip(_MARKS, bits) if on}
        if not (
            marks & _SPECIAL
            or (marks & {'unit', 'smoke', 'integration'} and 'slow' not in marks)
        ):
            continue
        for f in named + ['other.py']:
            n = sum(
                e.evaluate(marks.__contains__) for e, files in jobs if not files or f in files
            )
            if n != 1 and not (
                n == 0 and f == 'other.py' and accept.evaluate(marks.__contains__)
            ):
                faults.append((sorted(marks), f, n))
    return faults


@pytest.mark.unit
def test_the_nightly_jobs_select_each_marked_test_once():
    """Every combination of marks a test can carry runs in exactly one nightly job; every
    select is set and names only registered marks that the partition enumerates, and every
    named file exists."""
    import tomllib

    workflow = WORKFLOW.read_text()
    jobs = _jobs(workflow)
    assert all(sel for sel, _ in jobs)  # pytest -m '' would run every test
    names = {w for sel, _ in jobs for w in re.findall(r'\w+', sel)}
    assert names - {'and', 'or', 'not'} <= set(_MARKS)
    ini = tomllib.loads((ROOT / 'pyproject.toml').read_text())['tool']['pytest']['ini_options']
    assert set(_MARKS) <= {m.split(':')[0] for m in ini['markers']}
    assert all((ROOT / 'tests' / f).is_file() for _, files in jobs for f in files)
    assert _partition_faults(workflow) == []


@pytest.mark.unit
@pytest.mark.parametrize(
    'old, new',
    [
        pytest.param('select: paleos_adiabat', 'select: paleos_adiabatt', id='typo'),
        pytest.param('(unit or smoke or integration)', '(unit or smoke)', id='no_integration'),
        pytest.param(
            'suite: paleos_adiabat\n            select:',
            'suite: paleos_adiabat\n            selected:',
            id='no_paleos_job',
        ),
        pytest.param(
            'and not accept_solves and not accept_solves_long'
            ' and not accept_solves_newton_setups',
            'and not accept_solves',
            id='main_takes_sub_marks',
        ),
        pytest.param(
            'select: accept_solves and not accept_solves_long',
            'select: (accept_solves or unit) and not accept_solves_long',
            id='file_job_takes_unit',
        ),
        pytest.param(
            'select: accept_solves_long and not accept_solves_newton_setups and',
            'select: accept_solves_long and',
            id='long_takes_newton',
        ),
    ],
)
def test_a_gap_in_the_nightly_partition_is_found(old, new):
    """A typo in a select, a tier dropped from the main job, a job without a select or a job
    that takes another job's marks leaves a combination of marks in no job or in two."""
    workflow = WORKFLOW.read_text()
    assert old in workflow
    assert _partition_faults(workflow.replace(old, new))


@pytest.mark.unit
def test_every_file_with_the_accept_solves_mark_is_named_in_the_workflow():
    """A text scan with no collection: each test file that uses the accept_solves mark appears
    in nightly.yml, so a renamed or new file is not left out of the accept jobs."""
    workflow = WORKFLOW.read_text()
    marked = sorted(
        p.relative_to(ROOT).as_posix()
        for p in (ROOT / 'tests').rglob('test_*.py')
        if p != Path(__file__) and re.search(r'\bmark\.accept_solves\b', p.read_text())
    )
    assert 'tests/test_invariants.py' in marked and 'tests/test_accepted_on.py' in marked
    assert [m for m in marked if m not in workflow] == []


NEWTON_MARK = 'accept_solves_newton_setups'
SOLVES = ['test_pressure_root_solves.py']


# pytest-timeout applies the closest timeout marker of a test; this run prints it per test.
_TIMEOUTS = """
import sys, pytest
class Closest:
    def pytest_collection_finish(self, session):
        for item in session.items:
            m = item.get_closest_marker('timeout')
            print('TIMEOUT', item.nodeid, m.args[0] if m else None)
sys.exit(pytest.main(sys.argv[1:], plugins=[Closest()]))
"""


def _timeouts(select, keyword=None, files=()):
    """Closest timeout marker [s] of each test in ``files`` (the whole tree if none) that
    ``select`` and ``keyword`` take."""
    done = subprocess.run(
        [sys.executable, '-c', _TIMEOUTS, '--collect-only', '-q', '-m', select]
        + (['-k', keyword] if keyword else [])
        + ['-o', 'addopts=', '-p', 'no:cacheprovider', '--rootdir', str(ROOT / 'tests')]
        + list(files),
        cwd=ROOT / 'tests',
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert done.returncode in (0, 5), done.stdout[-2000:] + done.stderr[-2000:]
    rows = [
        line.split()[1:] for line in done.stdout.splitlines() if line.startswith('TIMEOUT ')
    ]
    return {node: None if t == 'None' else float(t) for node, t in rows}


def _newton_faults(workflow):
    """Faults of the jobs that take the Newton setup mark: each of its tests (in one file, see
    the smoke test below) runs in exactly one of them, each selects over the whole tree, no
    other job has a ``-k``, and the closest timeout markers of the tests of one job add up to
    less than its limit. The sum is the time of the tests run one after the other; they run
    at the same time on separate workers, so the rest of the sum is room for the job steps."""
    suites = _suites(workflow)
    entries = {n: _entry(job) for n, job in suites.items()}
    pattern = rf'(?<!not )\b{NEWTON_MARK}\b'
    split = [n for n, (select, _) in entries.items() if re.search(pattern, select)]
    faults = [('selects over named files', n) for n in split if entries[n][1]]
    faults += [
        ('-k outside the Newton jobs', n)
        for n, job in suites.items()
        if n not in split and _keyword(job) is not None
    ]
    marks = {
        n: _timeouts(entries[n][0], _keyword(suites[n]), entries[n][1] or SOLVES) for n in split
    }
    for node in sorted(_timeouts(NEWTON_MARK, files=SOLVES)):
        runs = [n for n, m in marks.items() if node in m]
        if len(runs) != 1:
            faults.append(('runs in', node, runs))
    for n, m in marks.items():
        if not m or None in m.values():
            faults.append(('no timeout marker', n, m))
        elif sum(m.values()) >= 60 * _job_limit(suites[n]):
            faults.append(('timeouts over the job limit', n, sum(m.values())))
    return faults


@pytest.mark.unit
def test_the_newton_setup_sweep_is_split_over_its_jobs():
    """The Newton setup sweep and the PALEOS solve within the cap are the longest real solves:
    each of their tests runs in one Newton job, and in the long job each closest timeout
    marker fires before the job limit. One matrix line sets the job limits."""
    workflow = WORKFLOW.read_text()
    assert _newton_faults(workflow) == []
    suites = _suites(workflow)
    assert len([n for n in suites if n.startswith(NEWTON_MARK)]) == 4
    assert not _entry(suites['accept_solves_long'])[1], 'the long job selects over the tree'
    key = r'^( *)["\']?timeout-minutes["\']?:\s*(.*?)\s*(?:#.*)?$'
    limits = re.findall(key, workflow, re.M)
    assert limits == [('    ', '${{ matrix.job_timeout || 90 }}')], limits
    marks = _timeouts(_entry(suites['accept_solves_long'])[0])
    assert marks and None not in marks.values(), marks
    assert max(marks.values()) < 60 * _job_limit(suites['accept_solves_long']), marks


@pytest.mark.unit
@pytest.mark.parametrize(
    'old, new, kind',
    [
        pytest.param(
            '"paleos_5_numpy or paleos_1_jax"', '"paleos_5_numpy"', 'runs in', id='dropped_id'
        ),
        pytest.param(
            '"within_the_cap"', '"within_the_cap or analytic_1"', 'runs in', id='id_in_2_jobs'
        ),
        pytest.param(
            '-k "paleos_1_numpy or paleos_3_jax"\n            job_timeout: 170',
            '-k "paleos_1_numpy or paleos_3_jax"\n            job_timeout: 120',
            'timeouts over the job limit',
            id='timeouts_over_limit',
        ),
        pytest.param(
            '-k "within_the_cap"',
            '-k "within_the_cap" tests/test_pressure_root_solves.py',
            'selects over named files',
            id='named_files',
        ),
        pytest.param(
            '-k "within_the_cap"', '-k "no_such_test"', 'no timeout marker', id='empty_job'
        ),
        pytest.param(
            '--durations=0 tests/test_pressure_root_solves.py',
            '--durations=0 -k "smooth_kink" tests/test_pressure_root_solves.py',
            '-k outside the Newton jobs',
            id='k_outside_newton',
        ),
        pytest.param(
            '--durations=0 tests/test_pressure_root_solves.py',
            "--durations=0 -k 'smooth_kink' tests/test_pressure_root_solves.py",
            '-k outside the Newton jobs',
            id='k_single_quotes',
        ),
        pytest.param(
            '--durations=0 tests/test_pressure_root_solves.py',
            '--durations=0 -k=smooth_kink tests/test_pressure_root_solves.py',
            '-k outside the Newton jobs',
            id='k_equals',
        ),
        pytest.param(
            'args: --dist load --cov-fail-under=0 --durations=0 tests/test_pressure_root_solves.py',
            'args: "--dist load --cov-fail-under=0 --durations=0 -k smooth_kink'
            ' tests/test_pressure_root_solves.py"',
            '-k outside the Newton jobs',
            id='k_in_quoted_args',
        ),
        pytest.param(
            '--durations=0 tests/test_pressure_root_solves.py',
            '--durations=0 -xk smooth_kink tests/test_pressure_root_solves.py',
            '-k outside the Newton jobs',
            id='k_grouped',
        ),
        pytest.param(
            '--durations=0 tests/test_pressure_root_solves.py',
            "--durations=0 -k smooth_kink tests/test_pressure_root_solves.py  # the kink's tests",
            '-k outside the Newton jobs',
            id='k_and_comment',
        ),
        pytest.param(
            '-k "within_the_cap"', '-k "within_the_cap" -k analytic_1', 'runs in', id='k_twice'
        ),
    ],
)
def test_a_fault_in_the_newton_split_is_found(old, new, kind):
    """A setup in no Newton job or in 2 of them, a job whose tests' timeouts add up to its
    limit or more, a Newton job that names files or runs no test, or a -k in any form on any
    other job is a fault."""
    workflow = WORKFLOW.read_text()
    assert workflow.count(old) == 1
    assert kind in {f[0] for f in _newton_faults(workflow.replace(old, new))}


@pytest.mark.unit
def test_each_newton_setup_timeout_covers_its_clock_and_the_retry():
    """The pytest timeout of each Newton setup ends after its own clock and the 600 s retry
    clock of ``main``, and a setup whose timeout equals its clock is found."""
    from tests.test_pressure_root_solves import NEWTON_SETUPS

    def short(setups):
        return [s[0] for s in setups if s[5] < s[4] + 600]

    assert short(NEWTON_SETUPS) == []
    assert short([s[:5] + (s[4],) for s in NEWTON_SETUPS]) == [s[0] for s in NEWTON_SETUPS]


@pytest.mark.smoke
def test_only_the_newton_setup_sweep_carries_its_mark():
    """The Newton setup jobs select over the whole tree (about 20 s to collect), and the
    mark is on one file only; the unit test above checks the tests of that file."""
    assert _marked(ROOT / 'tests', 'accept_solves_newton_setups') == {
        'test_pressure_root_solves.py'
    }


@pytest.mark.unit
def test_the_shared_wolf_bower_solves_run_once_in_the_wolfbower_job():
    """The wolfbower job spreads its tests by xdist group, and every test that asks for the
    module fixture wb_solves sits in the one class with the wb_solves group."""
    args = re.search(r'args:(.*)', _suites(WORKFLOW.read_text())['wolfbower']).group(1).split()
    assert args[args.index('--dist') + 1] == 'loadgroup'
    tree = ast.parse((ROOT / 'tests' / 'test_table_band_report.py').read_text())
    users = {
        getattr(cls, 'name', 'module')
        for cls in [tree, *(c for c in ast.walk(tree) if isinstance(c, ast.ClassDef))]
        for f in cls.body
        if isinstance(f, ast.FunctionDef) and 'wb_solves' in [a.arg for a in f.args.args]
    }
    assert users == {'TestWolfBowerSolves'}, users
    cls = next(c for c in tree.body if getattr(c, 'name', '') == 'TestWolfBowerSolves')
    groups = [ast.unparse(d) for d in cls.decorator_list if 'xdist_group' in ast.unparse(d)]
    assert groups == ["pytest.mark.xdist_group('wb_solves')"], groups
