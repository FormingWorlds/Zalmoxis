"""JAX against numpy on the inputs of the core temperature table cases.

Solves the 1 Earth-mass PALEOS planet of ``tests/test_core_temperature_arrays.py`` on the JAX
and on the numpy path, without and with the core table, at the ODE tolerance of its config
(``relative_tolerance`` 1e-5) and at 1e-8, and prints the relative difference of the radius,
the central pressure, the CMB pressure and the central density of the 2 paths.

Usage::

    python -m tools.validation.core_table_path_gap
"""

from __future__ import annotations

import numpy as np

from tests._core_table_helpers import _key, _solve, _table


def main():
    """Print the relative path difference of R, P_c, P_cmb and rho_c per case."""
    tight = {'relative_tolerance': 1e-8, 'absolute_tolerance': 1e-9}
    for name, table, ode in (
        ('no table, rtol 1e-5', None, {}),
        ('table, rtol 1e-5', _table, {}),
        ('table, rtol 1e-8', _table, tight),
    ):
        a = np.array(_key(_solve(jax=True, table=table, **ode)[0]))
        b = np.array(_key(_solve(jax=False, table=table, **ode)[0]))
        gap = ' '.join(f'{x:.2e}' for x in np.abs(a / b - 1.0))
        print(f'{name}: |JAX / numpy - 1| of R, P_c, P_cmb, rho_c = {gap}')


if __name__ == '__main__':
    main()
