"""JAX ports of Zalmoxis EOS inner kernels.

This subpackage holds jax.numpy / jax.jit reimplementations of the
performance-critical EOS kernels. The numpy implementations in
``zalmoxis.eos`` remain the reference; kernels here must match them
within solver-tolerance (rtol <= 1e-4 on all physics fields).

Module layout mirrors the numpy side:
    bilinear    — fast_bilinear_jax, paleos_clamp_temperature_jax
    paleos      — get_paleos_unified_density_jax
    rhs         — coupled_odes_jax RHS
    solver      — diffrax-based solve_structure replacement

The JAX path is gated behind the ``use_jax`` entry in ``config_params``;
when ``use_jax`` is False (the default) the numpy path is used unchanged.

JAX x64 mode is enabled at import. Without this, JAX defaults to float32
and all downstream density/pressure/temperature calculations lose ~1e-7
relative precision versus the numpy reference, far above the parity
tolerance (rtol <= 1e-12) we require for kernel-level parity.
"""

from __future__ import annotations

# Enable double-precision in JAX. Must happen before any jax.numpy op.
import jax as _jax

_jax.config.update('jax_enable_x64', True)
