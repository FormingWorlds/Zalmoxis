"""The nightly jobs that select ``accept_solves`` name their files; a marked test elsewhere
would run in no job. See docs/How-to/build_tests.md for the test tiers."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _marked(tests_dir):
    """Files under ``tests_dir``, as paths below it, with a test that pytest selects by
    ``-m accept_solves``."""
    done = subprocess.run(
        [sys.executable, '-m', 'pytest', '--collect-only', '-q', '-m', 'accept_solves']
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
    named = set()
    for job in re.split(r'\n\s*- suite:', workflow):
        select = re.search(r'select:(.*)', job)
        if select and re.search(r'(?<!not )\baccept_solves\b', select.group(1)):
            named |= set(re.findall(r'tests/([\w/]+\.py)', job))
    return sorted(marked - named)


@pytest.mark.unit
def test_every_accept_solves_file_runs_in_a_nightly_job():
    workflow = (ROOT / '.github' / 'workflows' / 'nightly.yml').read_text()
    marked = _marked(ROOT / 'tests')
    assert {'test_invariants.py', 'test_accepted_on.py'} <= marked
    assert _unselected(marked, workflow) == []


@pytest.mark.unit
def test_a_marked_file_outside_the_named_jobs_is_found(tmp_path):
    workflow = (ROOT / '.github' / 'workflows' / 'nightly.yml').read_text()
    for name in ('test_invariants.py', 'test_accepted_on.py', 'test_third.py'):
        (tmp_path / name).write_text(
            'import pytest\n\n@pytest.mark.accept_solves\ndef test_x(): pass\n'
        )
    (tmp_path / 'test_plain.py').write_text('def test_y(): pass\n')
    assert _unselected(_marked(tmp_path), workflow) == ['test_third.py']


_FORMS = {
    'list': 'import pytest\n\npytestmark = [\n    pytest.mark.unit,\n    pytest.mark.accept_solves,\n]\n\n\ndef test_x():\n    pass\n',
    'param': 'import pytest\n\n\n@pytest.mark.parametrize(\n    "a", [pytest.param(1, marks=pytest.mark.accept_solves)]\n)\ndef test_x(a):\n    pass\n',
    'alias': 'from pytest import mark\n\n\n@mark.accept_solves\ndef test_x():\n    pass\n',
    'sub/': 'import pytest\n\n\n@pytest.mark.accept_solves\ndef test_x():\n    pass\n',
}


@pytest.mark.unit
@pytest.mark.parametrize('form', sorted(_FORMS))
def test_every_form_of_the_marker_is_found(tmp_path, form):
    """A pytestmark list, a marked pytest.param, an aliased mark and a file in a subfolder all
    put a test in the accept_solves selection, so each is found when no job names it."""
    workflow = (ROOT / '.github' / 'workflows' / 'nightly.yml').read_text()
    name = 'sub/test_third.py' if form == 'sub/' else 'test_third.py'
    (tmp_path / name).parent.mkdir(exist_ok=True)
    (tmp_path / name).write_text(_FORMS[form])
    (tmp_path / 'test_plain.py').write_text('def test_y():\n    pass\n')
    assert _unselected(_marked(tmp_path), workflow) == [name]


@pytest.mark.unit
def test_a_marked_wolfbower_test_runs_in_the_wolfbower_job(tmp_path):
    """The accept_solves jobs select accept_solves and not wolfbower; a test with both marks
    runs in the wolfbower job, so it is not reported."""
    workflow = (ROOT / '.github' / 'workflows' / 'nightly.yml').read_text()
    (tmp_path / 'test_third.py').write_text(
        'import pytest\n\n\n@pytest.mark.accept_solves\n@pytest.mark.wolfbower\n'
        'def test_x():\n    pass\n'
    )
    assert _unselected(_marked(tmp_path), workflow) == []
