# How the adiabat is integrated

How does Zalmoxis turn the adiabatic gradient of an equation of state (EOS) table into a temperature profile, and how accurate is that profile?

In the temperature modes `adiabatic` and `adiabatic_from_cmb`, the temperature follows an adiabat: it rises with pressure as fast as the EOS allows without heat exchange.
`compute_adiabatic_temperature` in `zalmoxis.eos.temperature` integrates this adiabat from one node of the radial grid to the next.
It uses the midpoint rule in $\ln P$ with substeps of at most 0.0025 in $\ln P$, and it holds the temperature constant below 1 bar.
On the stored profiles of the default configuration and of the PROTEUS initial structure, the result lies within 0.1 K of a high-accuracy integration.

## The equation

The adiabatic gradient $\nabla_\mathrm{ad} = (\partial \ln T / \partial \ln P)_S$ is a dimensionless number of each EOS table; on the profiles below it lies between 0.009 and 0.35 at pressures above 1 GPa.
$T$ is the temperature in K, $P$ the pressure in Pa and $S$ the entropy.
With $x = \ln P$ and $y = \ln T$, an adiabat is the solution of

$$\frac{dy}{dx} = \nabla_\mathrm{ad}(P, T).$$

The pressure of a planet spans about 6 to 15 factors of $e$ (the profiles below span 5.9 to 15.2 in $\ln P$), so $x$ is the natural step variable.
The integration starts at an anchor: the surface temperature at the top node (`anchor='surface'`, inward), or the CMB temperature at the first mantle node (`anchor='cmb'`, outward through the mantle).
[Code architecture](code_architecture.md#adiabat-anchor-surface-vs-cmb) describes the 2 anchors.
For a mixture of materials, `get_mixed_nabla_ad` weights $\nabla_\mathrm{ad}$ of the components ([Multi-material mixing](mixing.md#adiabatic-gradient-mixing)); a segment takes the mixture of the node it fills.

## The midpoint step

Between 2 neighbouring nodes the routine takes $n = \lceil |\Delta x| / h \rceil$ equal substeps of size $\delta = \Delta x / n$, with $h$ = `max_dlnp` (default `ADIABAT_MAX_DLNP` = 0.0025). Each substep is

$$k_1 = \nabla_\mathrm{ad}(e^x, e^y), \qquad y \leftarrow y + \delta \, \nabla_\mathrm{ad}\left(e^{x + \delta/2}, e^{y + \delta k_1 / 2}\right).$$

The error of one substep is proportional to $\delta^3$, so the error of the whole profile falls as $h^2$ when $\nabla_\mathrm{ad}$ is smooth.
For a constant $\nabla_\mathrm{ad}$ the step is exact: $T = T_a (P / P_a)^{\nabla_\mathrm{ad}}$, with $T_a$ and $P_a$ the anchor values.

A worked example: the default configuration at a surface temperature of 2515 K has its centre at 407 GPa and its CMB at 141 GPa.
One call takes about 2400 substeps and reads $\nabla_\mathrm{ad}$ about 4800 times.
The centre then lies at 5487 to 5501 K over 4 starts.
One forward step in $P$ per node, a first-order scheme, puts the centre at 5765 to 5832 K, 278 to 330 K too hot.

## Accuracy on the PALEOS tables

PALEOS gives $\nabla_\mathrm{ad}$ by bilinear interpolation in $(\log_{10} P, \log_{10} T)$.
This value is continuous, but its slope jumps at every grid line of the table.
In the default core the adiabat also runs along the boundary between liquid and solid iron near 140 to 155 GPa, where $\nabla_\mathrm{ad}$ falls steeply.
A substep that crosses such a kink or a steep step loses part of its second-order accuracy, so on the real tables the error does not fall exactly as $h^2$.
The checks therefore use 2 tests: the order on smooth $\nabla_\mathrm{ad}$ with known solutions, and an error bound of 1 K on real profiles against an independent integration.

## Low pressure, padded tops and limits

- **Below 1 bar.** Both ends of a segment are raised to `ADIABAT_P_FLOOR` = $10^5$ Pa before the integration, so $T$ is constant below 1 bar. The PALEOS rock and iron tables start at $10^5$ Pa. For a solid or a liquid, $\nabla_\mathrm{ad} = \alpha P / (\rho c_p)$, with $\alpha$ the thermal expansivity, $\rho$ the density and $c_p$ the heat capacity, so $\nabla_\mathrm{ad}$ vanishes as $P \to 0$; for $\alpha = 3 \times 10^{-5}$ 1/K, $\rho$ = 3000 kg/m$^3$, $c_p$ = 1200 J/(kg K) and 2500 K, the adiabat from 0 to 1 bar changes $T$ by about 0.002 K. The temperature is continuous in the surface pressure.
- **Padded top.** A node with $P = 0$ (a [padded accept](process_flow.md#padded-accepts)) ends the integration: the top shell is isothermal.
- **Missing gradient.** Where $\nabla_\mathrm{ad}$ is missing or not positive, $T$ is held for the rest of that segment; the next segment integrates again.
- **Caps.** Each node is capped at $10^5$ K, the top of the PALEOS tables; with the CMB anchor it is also kept at 100 K or above.
- **Check.** Each call raises `StructureInvariantError` when $T$ is not finite and positive at every node, or when it rises outward from the anchor.
- **Range of `max_dlnp`.** A real number from $10^{-4}$ to 1; any other value raises `ValueError`.
- **Known limit.** A water layer, whose table reaches 0.1 Pa, is also isothermal below 1 bar.

**Where this is checked.** [Adiabat integration](../Validation/thermal_state.md#adiabat-integration-on-smooth-gradients) and the 2 cases after it on the Thermal state page.
