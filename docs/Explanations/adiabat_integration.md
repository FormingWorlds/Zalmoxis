# How the adiabat is integrated

How does Zalmoxis turn the adiabatic gradient of an equation of state (EOS) table into a temperature profile, and how accurate is that profile?

In the temperature modes `adiabatic` and `adiabatic_from_cmb`, the temperature follows an adiabat: the temperature profile of material that is compressed without heat exchange, so that its entropy stays constant.
`compute_adiabatic_temperature` in `zalmoxis.eos.temperature` integrates this adiabat from one node of the radial grid to the next.
It uses the midpoint rule in $\ln P$ with substeps of at most 0.0025 in $\ln P$, and it holds the temperature constant below 1 bar.
On the stored profiles of the default configuration and of the PROTEUS initial structure, the result lies within 0.54 K of an independent high-accuracy integration on macOS ([the values per profile and platform](../Validation/thermal_state.md#adiabat-integration-on-real-profiles)); the test bound is 1 K.

## The equation

The adiabatic gradient $\nabla_\mathrm{ad} = (\partial \ln T / \partial \ln P)_S$ is dimensionless. The PALEOS tables hold it as a column. For `WolfBower2018:MgSiO3` and `RTPress100TPa:MgSiO3`, Zalmoxis computes it from the tabulated melt gradient $(dT/dP)_S$ as $\nabla_\mathrm{ad} = (dT/dP)_S\,P/T$ ([Equations of state](eos_physics.md#wolf-bower-2018-temperature-dependent-eos)). On the 6 stored profiles of the [real-profile check](../Validation/thermal_state.md#adiabat-integration-on-real-profiles) it lies between 0.009 and 0.43 at pressures above 1 GPa.
$T$ is the temperature in K, $P$ the pressure in Pa and $S$ the entropy.
With $x = \ln P$ and $y = \ln T$, an adiabat is the solution of

$$\frac{dy}{dx} = \nabla_\mathrm{ad}(P, T).$$

Between the anchor and the last integrated node, $\ln P$ changes by 9.9 to 15.2 on these profiles. Equal steps in $x$ give the same relative change of $P$ in every step, so $x$ is the natural step variable.
The integration starts at an anchor: the surface temperature at the top node (`anchor='surface'`, inward), or the CMB temperature at the first mantle node (`anchor='cmb'`, outward through the mantle). With the CMB anchor, the nodes below the CMB keep the surface temperature.
[Code architecture](code_architecture.md#adiabat-anchor-surface-vs-cmb) describes the 2 anchors.
For a mixture of materials, `get_mixed_nabla_ad` weights $\nabla_\mathrm{ad}$ of the components ([Multi-material mixing](mixing.md#adiabatic-gradient-mixing)); a segment takes the mixture of the node it fills.

## The midpoint step

Between 2 neighbouring nodes the routine takes $n = \lceil |\Delta x| / h \rceil$ equal substeps of size $\delta = \Delta x / n$, with $h$ = `max_dlnp` (default `ADIABAT_MAX_DLNP` = 0.0025). Each substep is

$$k_1 = \nabla_\mathrm{ad}(e^x, e^y), \qquad y \leftarrow y + \delta \, \nabla_\mathrm{ad}\left(e^{x + \delta/2}, e^{y + \delta k_1 / 2}\right).$$

The error of one substep is proportional to $\delta^3$, so the error of the whole profile falls as $h^2$ when $\nabla_\mathrm{ad}$ is smooth.
For a constant $\nabla_\mathrm{ad}$ the step is exact: $T = T_a (P / P_a)^{\nabla_\mathrm{ad}}$, with $T_a$ and $P_a$ the anchor values.

A worked example: the default configuration at a surface temperature of 2515 K has its centre at 407 GPa and its CMB at 141 GPa.
One call takes about 6200 substeps and reads $\nabla_\mathrm{ad}$ about 12300 times.
The test stores 4 starts of this configuration. Their planet masses differ only by $k \times 10^{-12}$ of the value ($k$ = 0 to 3), and their centre lies at 6238 to 6239 K.
A forward Euler step, one first-order step in $P$ per node that also holds a padded top shell isothermal, puts the centre at 5765 to 5832 K, 407 to 473 K too cold.

## Accuracy on the PALEOS tables

PALEOS gives $\nabla_\mathrm{ad}$ by bilinear interpolation in $(\log_{10} P, \log_{10} T)$.
This value is continuous, but its slope jumps at every grid line of the table.
In the default core the adiabat also crosses the boundary between liquid and solid iron near 220 to 260 GPa, where $\nabla_\mathrm{ad}$ changes steeply.
A substep that crosses such a kink or a steep step loses part of its second-order accuracy, so on the real tables the error does not fall exactly as $h^2$.
The checks therefore use 2 tests: the order on smooth $\nabla_\mathrm{ad}$ with known solutions, and an error bound of 1 K on real profiles against an independent integration.

## Low pressure, padded tops and limits

- **Below 1 bar.** Both ends of a segment are raised to `ADIABAT_P_FLOOR` = $10^5$ Pa before the integration, so $T$ is constant below 1 bar. The PALEOS rock and iron tables start at $10^5$ Pa. For a solid or a liquid, $\nabla_\mathrm{ad} = \alpha P / (\rho c_p)$, with $\alpha$ the thermal expansivity, $\rho$ the density and $c_p$ the heat capacity, so $\nabla_\mathrm{ad}$ vanishes as $P \to 0$; for $\alpha = 3 \times 10^{-5}$ 1/K, $\rho$ = 3000 kg/m$^3$, $c_p$ = 1200 J/(kg K) and 2500 K, the adiabat from 0 to 1 bar changes $T$ by about 0.002 K. The temperature is continuous in the top pressure, down to a padded top.
- **Padded top.** A node with $P = 0$ (a [padded accept](process_flow.md#padded-accepts)) counts as $10^5$ Pa, so a padded top gives the profile of a top at 1 bar.
- **Missing gradient.** Where $\nabla_\mathrm{ad}$ is missing or not positive, $T$ is held for the rest of that segment; the next segment integrates again.
- **Caps.** No node and no substep goes above $10^5$ K, the top of the PALEOS tables. With the CMB anchor, no node and no substep goes below 100 K either. A very large or infinite $\nabla_\mathrm{ad}$ therefore gives nodes at exactly $10^5$ K with the surface anchor, and nodes at exactly 100 K outward from the CMB anchor.
- **Range of the anchors.** `compute_adiabatic_temperature` raises `ValueError` for an anchor temperature outside its range, a bool included:
    - the surface temperature: a real number above 0 K and at most $10^5$ K;
    - with `anchor='cmb'`, the CMB temperature: from 100 K to $10^5$ K.

    The config check `validate_config` refuses these values before a solve starts. In every temperature mode it refuses a surface temperature that is not a positive, finite number. In the modes `adiabatic` and `adiabatic_from_cmb` it also refuses a surface temperature above $10^5$ K, and in `adiabatic_from_cmb` a CMB temperature outside 100 K to $10^5$ K. `solver.main` does not call `validate_config`, and PROTEUS calls `solver.main` with its own configuration dict, so on that path the anchors are checked only when the solve computes the adiabat itself, at its first call of `compute_adiabatic_temperature`.

- **Surface anchor below 100 K.** The routine accepts it, but the solver floors the temperature that the density reads from the adiabat at 100 K, so below 100 K the density uses 100 K.
- **Check.** Each call raises `StructureInvariantError` when $T$ is not finite and positive at every node, or when it rises outward from the anchor.
- **Range of `max_dlnp`.** A real number from $10^{-4}$ to 1; any other value raises `ValueError`.
- **Known limit.** A water layer, whose table reaches 0.1 Pa, is also isothermal below 1 bar.

**Where this is checked.** [Adiabat integration on smooth gradients](../Validation/thermal_state.md#adiabat-integration-on-smooth-gradients), [Adiabat integration on real profiles](../Validation/thermal_state.md#adiabat-integration-on-real-profiles) and [Temperature across 1 bar](../Validation/thermal_state.md#temperature-across-1-bar). Unit tests check the anchor ranges and the caps: `tests/test_adiabatic.py::TestMidpointAdiabat::test_anchor_temperature_property_over_its_domain`, `::test_a_huge_nabla_stops_at_the_node_caps` and `::test_a_segment_stops_exactly_at_its_limit_property` for the routine, and `tests/test_config_validation.py::TestTemperatureValidation::test_anchor_temperature_property_over_its_domain` for the config check.
