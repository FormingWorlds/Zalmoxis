"""A default-configuration solve is bit-identical to the main commit the two-layer core
branched from (d47933faa59fb31c3d8c951ab77a0364c031ec28): every array and scalar of the result,
against a reference that ``tools/default_solve_reference.py`` wrote on that commit, one per
platform (macOS arm64 locally, Linux x86_64 on the runner). A platform without a reference
fails and says so; a change of numpy, scipy or the data needs a new reference from that commit.
"""

from __future__ import annotations

import importlib.util
import platform
import sys
from pathlib import Path

import numpy as np
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(1800)]

ROOT = Path(__file__).resolve().parents[1]
KEY = f'{sys.platform}-{platform.machine()}'
REF = ROOT / 'tests' / 'data' / f'default_solve_reference_{KEY}.npz'


def test_a_default_solve_is_bit_identical_to_main():
    if not REF.is_file():
        pytest.fail(f'no default-solve reference for {KEY}: {REF.name} is missing')
    spec = importlib.util.spec_from_file_location(
        'ref', ROOT / 'tools' / 'default_solve_reference.py'
    )
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    got, ref = tool.solve_arrays(), np.load(REF)
    assert set(got) == set(ref.files)
    differ = [k for k in ref.files if got[k].tobytes() != ref[k].tobytes()]
    assert not differ, f'entries not bit-identical to main: {differ}'
