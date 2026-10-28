"""The per-file nightly jobs that select ``accept_solves`` name their files, so a marked test
elsewhere would run in no job; the ``accept_solves_long`` and ``accept_solves_newton_setups``
jobs select over the whole tree. See docs/How-to/build_tests.md for the test tiers."""

from __future__ import annotations

import ast
import itertools
import re
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


def _jobs(workflow):
    """(select, test files named in the job) of each nightly suite."""
    return [_entry(job) for job in _suites(workflow).values()]


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


NEWTON = (
    'test_pressure_root_solves.py::test_every_newton_setup_meets_the_surface_boundary_condition'
)


def _solves_nodes(select, files):
    """Node ids of tests/test_pressure_root_solves.py that a job with ``select`` runs."""
    if files and 'test_pressure_root_solves.py' not in files:
        return set()
    done = subprocess.run(
        [sys.executable, '-m', 'pytest', '--collect-only', '-q', '-m', select]
        + ['-o', 'addopts=', '-p', 'no:cacheprovider', '--rootdir', str(ROOT / 'tests')]
        + ['test_pressure_root_solves.py'],
        cwd=ROOT / 'tests',
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert done.returncode in (0, 5), done.stdout[-2000:] + done.stderr[-2000:]
    return {line for line in done.stdout.splitlines() if '::' in line}


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


def _timeouts(select):
    """Closest timeout marker [s] of each test in the whole tree that ``select`` takes."""
    done = subprocess.run(
        [sys.executable, '-c', _TIMEOUTS, '--collect-only', '-q', '-m', select]
        + ['-o', 'addopts=', '-p', 'no:cacheprovider', '--rootdir', str(ROOT / 'tests')],
        cwd=ROOT / 'tests',
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert done.returncode == 0, done.stdout[-2000:] + done.stderr[-2000:]
    rows = [
        line.split()[1:] for line in done.stdout.splitlines() if line.startswith('TIMEOUT ')
    ]
    return {node: None if t == 'None' else float(t) for node, t in rows}


@pytest.mark.unit
def test_the_newton_setup_sweep_runs_alone_in_its_own_job():
    """The sweep over every Newton setup is the longest real solve: it runs in one job, alone,
    and the closest timeout marker of each test in the long and Newton setup jobs fires before
    the job limit, so a slow runner names the test."""

    suites = _suites(WORKFLOW.read_text())
    nodes = {name: _solves_nodes(*_entry(job)) for name, job in suites.items()}
    assert [name for name, ids in nodes.items() if NEWTON in ids] == [
        'accept_solves_newton_setups'
    ]
    assert nodes['accept_solves_newton_setups'] == {NEWTON}
    assert (
        not _entry(suites['accept_solves_newton_setups'])[1]
        + _entry(suites['accept_solves_long'])[1]
    ), 'the long and Newton setup jobs select over the whole tree'
    key = r'^( *)["\']?timeout-minutes["\']?:\s*(.*?)\s*(?:#.*)?$'
    limits = re.findall(key, WORKFLOW.read_text(), re.M)
    assert limits == [('    ', '${{ matrix.job_timeout || 90 }}')], limits
    for name in ('accept_solves_newton_setups', 'accept_solves_long'):
        marks = _timeouts(_entry(suites[name])[0])
        assert marks and None not in marks.values(), (name, marks)
        assert max(marks.values()) < 60 * _job_limit(suites[name]), (name, marks)


@pytest.mark.smoke
def test_only_the_newton_setup_sweep_carries_its_mark():
    """The Newton setup job selects over the whole tree (about 20 s to collect), and the
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
