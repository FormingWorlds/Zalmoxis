# JAX EOS subpackage

The `zalmoxis.jax_eos` subpackage holds line-by-line JAX ports of the performance-critical EOS kernels and the structure ODE driver. The numpy implementations in `zalmoxis.eos` remain the reference; kernels here must match them within solver tolerance (rtol $\leq 10^{-12}$ on bilinear, rtol $\leq 10^{-6}$ on the full RHS, rtol $\leq 10^{-5}$ end-to-end at default integrator tolerance).

The path is gated by `config_params['use_jax']`; when False (the default) none of these modules are imported. Float64 is enforced at import via `jax.config.update('jax_enable_x64', True)` because JAX defaults to float32 and would otherwise lose ~$10^{-7}$ precision relative to the numpy reference. Scope today is the Stage-1b two-layer config (PALEOS:iron core + PALEOS-2phase:MgSiO$_3$ mantle); other configurations transparently fall back to the numpy path at the caller.

| Submodule | Purpose |
|---|---|
| `bilinear` | `fast_bilinear_jax`, `paleos_clamp_temperature_jax` |
| `paleos` | `get_paleos_unified_density_jax`, mushy-zone branches |
| `tdep` | `get_Tdep_density_jax` for the PALEOS-2phase mantle |
| `rhs` | `coupled_odes_jax` (jax-traceable structure RHS) |
| `solver` | `solve_structure_jax` via `diffrax.diffeqsolve(Tsit5)` with event-based pressure-zero termination |
| `wrapper` | `solve_structure_via_jax` (numpy-signature entry point used by `structure_model.solve_structure`); accepts both `temperature_function(r, P)` and an explicit `temperature_arrays = (r_arr, T_arr)` |

::: zalmoxis.jax_eos
    options:
      members: true
      inherited_members: false
      show_source: true
