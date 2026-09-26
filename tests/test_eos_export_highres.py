"""Parity of the loader with the genfromtxt reader on the 600-700 MB highres PALEOS tables.

Each table takes about a minute and several GB of memory, so the test is in the
slow tier and needs the highres files under ``FWL_DATA`` or the data directory.
"""

from __future__ import annotations

import pytest

from tests.test_eos_export import _assert_table_matches_genfromtxt, _real_paleos_tables

pytestmark = pytest.mark.slow


@pytest.mark.parametrize('path', _real_paleos_tables(highres=True), ids=lambda p: p.name)
def test_highres_tables_are_identical_to_the_genfromtxt_reader(path):
    """Every grid cell, phase label and empty cell equals the genfromtxt reading."""
    _assert_table_matches_genfromtxt(path)
