"""A default-configuration solve is bit-identical to the one-layer core before the two-layer
core: every array and scalar of the result, against a reference that
``tools/default_solve_reference.py`` wrote with the one-layer code, one per platform (macOS
arm64 locally, Linux x86_64 on the runner). A platform without a reference fails and says so;
a change of numpy, scipy or the data needs a new reference from the one-layer code.
"""

from __future__ import annotations

import platform
import sys
from pathlib import Path

import numpy as np
import pytest

from tools.default_solve_reference import solve_arrays

pytestmark = [pytest.mark.integration, pytest.mark.timeout(1800)]

ROOT = Path(__file__).resolve().parents[1]
KEY = f'{sys.platform}-{platform.machine()}'
REF = ROOT / 'tests' / 'data' / f'default_solve_reference_{KEY}.npz'


def test_a_default_solve_is_bit_identical_to_main():
    if not REF.is_file():
        pytest.fail(f'no default-solve reference for {KEY}: {REF.name} is missing')
    got, ref = solve_arrays(), np.load(REF)
    assert set(got) == set(ref.files)
    differ = [k for k in ref.files if got[k].tobytes() != ref[k].tobytes()]
    assert not differ, f'entries not bit-identical to main: {differ}'
