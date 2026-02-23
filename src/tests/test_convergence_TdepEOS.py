from __future__ import annotations

import os

import pytest

from tools.setup_tests import run_zalmoxis_TdepEOS

# Read the environment variable for ZALMOXIS_ROOT
ZALMOXIS_ROOT = os.getenv('ZALMOXIS_ROOT')
if not ZALMOXIS_ROOT:
    raise RuntimeError('ZALMOXIS_ROOT environment variable not set')


@pytest.mark.parametrize(
    'mass', [1, 2, 3, 4, 5, 6]
)  # 1 to 6 Earth masses (keep it simple for CI tests; works up to 6.7)
def test_all_compositions_converge(mass):
    """ """
    print(f'Running test for mass = {mass}')

    # Delete composition_TdepEOS_mass_log file if it exists
    custom_log_file = os.path.join(
        ZALMOXIS_ROOT, 'output_files', 'composition_TdepEOS_mass_log.txt'
    )
    if os.path.exists(custom_log_file):
        os.remove(custom_log_file)

    # Run Zalmoxis for a given mass with temperature-dependent EOS
    results = run_zalmoxis_TdepEOS(mass)

    # Filter out any failed convergence
    failed_cases = [(id_mass) for (id_mass, converged) in results if not converged]

    if failed_cases:
        failed_str = ', '.join([f'mass={id_mass:.2f}' for id_mass in failed_cases])
        pytest.fail(f'The following masses did not converge {mass}: {failed_str}')
