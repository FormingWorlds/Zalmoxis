"""The nightly jobs that select ``accept_solves`` name their files; a marked test elsewhere
would run in no job. See docs/How-to/build_tests.md for the test tiers."""

from __future__ import annotations

import itertools
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


WORKFLOW = ROOT / '.github' / 'workflows' / 'nightly.yml'


def _jobs(workflow):
    """(select, test files named in the job) of each nightly suite."""
    out = []
    for job in re.split(r'\n\s*- suite:', workflow)[1:]:
        select = re.search(r'select:(.*)', job)
        out.append(
            (select.group(1).strip() if select else '', re.findall(r'tests/([\w/]+\.py)', job))
        )
    return out


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
            'accept_solves and not wolfbower', 'accept_solves', id='accept_takes_wolfbower'
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
)
_SPECIAL = {'wolfbower', 'paleos_adiabat', 'accept_solves'}


def _partition_faults(workflow):
    """Marker sets, with the file for accept_solves, that no job or more than one job selects.
    A set counts when it has a special mark, or unit, smoke or integration without slow."""
    from _pytest.mark.expression import Expression

    jobs = [(Expression.compile(sel), files) for sel, files in _jobs(workflow)]
    named = sorted({f for _, files in jobs for f in files})
    faults = []
    for bits in itertools.product((False, True), repeat=len(_MARKS)):
        marks = {m for m, on in zip(_MARKS, bits) if on}
        if not (
            marks & _SPECIAL
            or (marks & {'unit', 'smoke', 'integration'} and 'slow' not in marks)
        ):
            continue
        for f in named if 'accept_solves' in marks else ['other.py']:
            n = sum(
                e.evaluate(marks.__contains__) for e, files in jobs if not files or f in files
            )
            if n != 1:
                faults.append((sorted(marks), f, n))
    return faults


@pytest.mark.unit
def test_the_nightly_jobs_select_each_marked_test_once():
    """Every combination of marks a test can carry runs in exactly one nightly job."""
    assert _partition_faults(WORKFLOW.read_text()) == []


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
    ],
)
def test_a_gap_in_the_nightly_partition_is_found(old, new):
    """A typo in a select, a tier dropped from the main job or a job without a select leaves
    a combination of marks in no job."""
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
