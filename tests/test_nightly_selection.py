"""The nightly jobs that select ``accept_solves`` name their files; a marked test elsewhere
would run in no job. See docs/How-to/build_tests.md for the test tiers."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _unselected(tests_dir, workflow):
    """Files under ``tests_dir`` that carry the accept_solves marker but are named in no
    nightly job whose select takes accept_solves."""
    named = set()
    for job in re.split(r'\n\s*- suite:', workflow):
        select = re.search(r'select:(.*)', job)
        if select and re.search(r'(?<!not )\baccept_solves\b', select.group(1)):
            named |= set(re.findall(r'tests/(test_\w+\.py)', job))
    mark = re.compile(r'^\s*(@pytest\.mark\.accept_solves\b|pytestmark\b.*accept_solves)', re.M)
    marked = {p.name for p in tests_dir.glob('test_*.py') if mark.search(p.read_text())}
    return sorted(marked - named)


@pytest.mark.unit
def test_every_accept_solves_file_runs_in_a_nightly_job():
    workflow = (ROOT / '.github' / 'workflows' / 'nightly.yml').read_text()
    assert _unselected(ROOT / 'tests', workflow) == []


@pytest.mark.unit
def test_a_marked_file_outside_the_named_jobs_is_found(tmp_path):
    workflow = (ROOT / '.github' / 'workflows' / 'nightly.yml').read_text()
    for name in ('test_invariants.py', 'test_accepted_on.py', 'test_third.py'):
        (tmp_path / name).write_text(
            'import pytest\n\n@pytest.mark.accept_solves\ndef test_x(): pass\n'
        )
    (tmp_path / 'test_plain.py').write_text('def test_y(): pass\n')
    assert _unselected(tmp_path, workflow) == ['test_third.py']
