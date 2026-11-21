# How the adiabat is integrated

How does Zalmoxis turn the adiabatic gradient of an equation of state (EOS) table into a temperature profile, and how accurate is that profile?

In the temperature modes `adiabatic` and `adiabatic_from_cmb`, the temperature follows an adiabat: the temperature profile of material that is compressed without heat exchange, so that its entropy stays constant.
`compute_adiabatic_temperature` in `zalmoxis.eos.temperature` integrates this adiabat from one node of the radial grid to the next.
It uses the midpoint rule in $\ln P$ with substeps of at most 0.0025 in $\ln P$, and it holds the temperature constant below 1 bar.
On the stored profiles of the default configuration and of the PROTEUS initial structure, the result lies within 0.254 K of an independent high-accuracy integration on macOS ([the values per profile](../Validation/thermal_state.md#adiabat-integration-on-real-profiles)); the test bound is 1 K.

## The equation

$T$ is the temperature in K, $P$ the pressure in Pa and $S$ the entropy. The adiabatic gradient $\nabla_\mathrm{ad} = (\partial \ln T / \partial \ln P)_S$ is dimensionless. With $x = \ln P$ and $y = \ln T$, an adiabat is the solution of

$$\frac{dy}{dx} = \nabla_\mathrm{ad}(P, T).$$

Equal steps in $x$ give the same relative change of $P$ in every step, so $x$ is the natural step variable. Between the anchor and the last integrated node, $\ln P$ changes by 9.9 to 15.2 on the 6 stored profiles of the [real-profile check](../Validation/thermal_state.md#adiabat-integration-on-real-profiles).

**Where the gradient comes from.** The PALEOS tables hold $\nabla_\mathrm{ad}$ as a column. For `WolfBower2018:MgSiO3` and `RTPress100TPa:MgSiO3`, Zalmoxis computes it from the tabulated melt gradient $(dT/dP)_S$ as $\nabla_\mathrm{ad} = (dT/dP)_S\,P/T$ ([Equations of state](eos_physics.md#wolf-bower-2018-temperature-dependent-eos)). On the 6 stored profiles it lies between 0.009 and 0.43 at the nodes above 1 GPa, and between 0.0057 and 0.433 at all reads of the substeps above 1 GPa. For a mixture of materials, `get_mixed_nabla_ad` weights $\nabla_\mathrm{ad}$ of the components ([Multi-material mixing](mixing.md#adiabatic-gradient-mixing)); a segment takes the mixture of the node it fills.

**Where the integration starts.** The integration starts at an anchor: the surface temperature at the top node (`anchor='surface'`, inward), or the CMB temperature at the first mantle node (`anchor='cmb'`, outward through the mantle). With the CMB anchor, the nodes below the CMB keep the surface temperature. [Code architecture](code_architecture.md#adiabat-anchor-surface-vs-cmb) describes the 2 anchors.

## The midpoint step

Between 2 neighbouring nodes the routine takes $n = \lceil |\Delta x| / h \rceil$ equal substeps of size $\delta = \Delta x / n$, with $h$ = `max_dlnp` (default `ADIABAT_MAX_DLNP` = 0.0025). Each substep is

$$k_1 = \nabla_\mathrm{ad}(e^x, e^y), \qquad y \leftarrow y + \delta \, \nabla_\mathrm{ad}\left(e^{x + \delta/2}, e^{y + \delta k_1 / 2}\right).$$

The error of one substep is proportional to $\delta^3$, so the error of the whole profile falls as $h^2$ when $\nabla_\mathrm{ad}$ is smooth.
For a constant $\nabla_\mathrm{ad}$ the step is exact: $T = T_a (P / P_a)^{\nabla_\mathrm{ad}}$, with $T_a$ and $P_a$ the anchor values.

A worked example: the default configuration at a surface temperature of 2515 K has its centre at 407 GPa and its CMB at 141 GPa.
One call takes about 6200 substeps and reads $\nabla_\mathrm{ad}$ about 12300 times.
The real-profile test stores 4 starts of this configuration. Their planet masses differ only by $k \times 10^{-12}$ of the value ($k$ = 0 to 3), and their centre lies at 6163 to 6168 K.

## Accuracy on the PALEOS tables

PALEOS gives $\nabla_\mathrm{ad}$ by bilinear interpolation in $(\log_{10} P, \log_{10} T)$.
This value is continuous, but its slope jumps at every grid line of the table.
In the default core the adiabat also runs along the melting curve of the PALEOS iron table, its boundary between liquid and solid iron: on the 4 stored default profiles the integrated path switches between liquid and solid cells of the table 5 times from 226 to 252 GPa, and $\nabla_\mathrm{ad}$ changes steeply there.
A substep that crosses such a kink or a steep step loses part of its second-order accuracy, so on the real tables the error does not fall exactly as $h^2$.
It also does not fall with every smaller $h$: on the 6 stored profiles the largest error against the independent integration is 0.254 K at $h = 0.0025$, 1.29 K at 0.0055, 1.25 K at 0.0065 and 2.34 K at 0.007.
The 1 K bound of the real-profile test is a measurement on these 6 profiles, with a margin of about 4 at the default $h$, not a bound for every profile.
The checks therefore use 2 tests: the order on smooth $\nabla_\mathrm{ad}$ with known solutions, and an error bound of 1 K on real profiles against an independent integration.

## Low pressure, padded tops, missing gradients and caps

- **Below 1 bar.** Both ends of a segment are raised to `ADIABAT_P_FLOOR` = $10^5$ Pa before the integration, so $T$ is constant below 1 bar. The PALEOS rock and iron tables start at $10^5$ Pa. For a solid or a liquid, $\nabla_\mathrm{ad} = \alpha P / (\rho c_p)$, with $\alpha$ the thermal expansivity, $\rho$ the density and $c_p$ the heat capacity, so $\nabla_\mathrm{ad}$ vanishes as $P \to 0$; for $\alpha = 3 \times 10^{-5}$ 1/K, $\rho$ = 3000 kg/m$^3$, $c_p$ = 1200 J/(kg K) and 2500 K, the adiabat from 0 to 1 bar changes $T$ by about 0.002 K. The temperature is continuous in the top pressure, down to a padded top.
- **Padded top.** A node with $P = 0$ (a [padded accept](process_flow.md#padded-accepts)) counts as $10^5$ Pa, so a padded top gives the profile of a top at 1 bar.
- **Missing gradient.** Where $\nabla_\mathrm{ad}$ is missing or not positive, $T$ is held for the rest of that segment; the next segment integrates again.
- **Caps.** No node and no substep goes above $10^5$ K, the top of the PALEOS tables. With the CMB anchor, no node and no substep goes below 100 K either. A very large or infinite $\nabla_\mathrm{ad}$ therefore gives nodes at exactly $10^5$ K with the surface anchor, and nodes at exactly 100 K outward from the CMB anchor.
- **A padded node with the CMB anchor.** In `adiabatic_from_cmb` a padded node takes its temperature from `_adiabat_to_surface` in `solver.py`, not from this routine: 64 midpoint steps from the last node with positive pressure to the target surface pressure, with no 1 bar floor. From 5 GPa and 3500 K on the `dummy` mantle it gives 0.113 K more than the segment rule at 1 atm and 0.457 K more at 1 Pa.

## How the density solve reads the temperature over the top cell

Over the top cell the density solve reads the adiabat at 64 pressures, not a straight line between its 2 ends.

The density solve reads $T$ from a table of this adiabat against $\log_{10} P$.
The solve builds the table on the nodes of the previous outer pass ([Full-adiabat rule](process_flow.md#full-adiabat-rule)).
The top cell is the pressure interval between the top of the planet and the first node below it; no node lies inside it.
Its ends depend on the top:

- **Padded top:** from 1 bar to the outermost node with a positive pressure.
- **Unpadded top at or above 1 bar:** from the top node to the next node.
- **Unpadded top below 1 bar:** from 1 bar to the next node; the top node keeps its place in the table.

The solve adds 64 pressures over the cell to the table, evenly spaced in $\log P$: the shell.
On the shell, the code integrates the adiabat inward from the temperature of the top node, for both anchors.
Over an unpadded cell each shell point takes its enclosed mass $m$ from $P$ linear in $m$ between the 2 ends, so the shell changes layer where the adiabat cuts the cell ([the ice boundary](#the-ice-boundary)).
Over a padded cell, which the adiabat does not cut, the shell takes the mixture of the deeper node of the cell with the surface anchor and of the first padded node with the CMB anchor.

On the stored profiles the cell spans 2 to 4 decades in pressure.
A straight line in $\log_{10} P$ over it lies 9 to 14 K off on the 2 PROTEUS profiles and up to 28 K off on the default start with its top node near 1 bar; the lookup lies at most 0.083 K from an independent integration.

**Where this is checked.** [Temperature lookup over the top cell](../Validation/thermal_state.md#temperature-lookup-over-the-top-cell).

## Layer boundaries inside a node segment

What does the routine do when the inner-core boundary (ICB), the core-mantle boundary (CMB) or the boundary between the mantle and an ice layer lies between 2 grid nodes?

Each node takes its temperature from the integration over the node segment from its neighbour on the anchor side.
When a boundary lies inside a segment, the routine cuts the segment there and integrates each piece (between 2 cuts, or a cut and a node) with the mixture of its layer.
The cut lies at the pressure of the boundary:

- **CMB:** the pressure at the core mass, with $\ln P$ taken linear in the enclosed mass $m$ between the 2 nodes.
- **Ice boundary:** the pressure at the core-plus-mantle mass, with $P$ linear in $m$ between the 2 nodes ([the ice boundary](#the-ice-boundary)).
- **ICB:** the pressure that the [inner-core boundary rule](mixing.md#the-inner-core-boundary) gives for the inner-core mass, with $P$ taken linear in $m^{2/3}$ between the core nodes on either side of it. In the node segment of the CMB, the CMB point takes the place of the upper node ([the pressure at the inner-core boundary](mixing.md#the-pressure-at-the-inner-core-boundary)).

So the cut and the ICB rule take one pressure for one inner-core mass, unless the ICB, the CMB and the ice boundary share one segment (the CMB point then takes $P$ linear in $m$).
Other segments take the mixture of the node they fill.

At the ICB of a pure-iron core at 258 GPa and 5537 K (on the Anzellini curve), $\nabla_\mathrm{ad}$ of the default side tables is 0.22 on the solid side and 0.44 on the liquid side.
Without the cut, the segment would take one of the 2 values, and the first inner-core node would miss the adiabat by up to 21 K.

**Where this is checked.** [Temperature at the CMB and the centre](../Validation/thermal_state.md#temperature-at-the-cmb-and-the-centre) for the CMB and [The adiabat across the ICB](../Validation/hydrostatic_structure.md#the-adiabat-across-the-icb) for the ICB.

### The ice boundary

Why do the CMB and the ice boundary take different rules?
In short: a top cell spans up to 4 decades in $P$, and there $\ln P$ linear in $m$ puts the boundary pressure far too low; a segment at the CMB spans about 0.03 in $\ln P$, where the 2 rules agree.
$P$ linear in $m$ gives equal pressure steps for equal mass steps, $\ln P$ linear in $m$ equal pressure factors.

In hydrostatic equilibrium $\mathrm{d}P/\mathrm{d}m = -G m / (4 \pi r^4)$, and over one segment $m$ and $r$ change little, so $P$ is close to linear in $m$.
On the default start with its top at $1.2 \times 10^5$ Pa, $\ln P$ linear in $m$ puts the pressure at the middle mass of the top cell at 12 MPa, and $P$ linear in $m$ at 0.58 GPa.
With $P$ linear in $m$ that pressure lies 0.44 % above the integral of $\mathrm{d}P/\mathrm{d}m$ inside the cell.

When the CMB lies in the same segment, its cut also takes $P$ linear in $m$, so the 2 cuts keep their order and the mantle between them keeps its piece.
A piece that ends on a node on the boundary takes the mixture of its middle, so an outward step into that node stays in the mantle.
The T(P) table of the density solve gains the boundary as one more point, with the temperature of the adiabat there, so its lookup does not interpolate across the change of slope.

A layer whose EOS has no temperature dependence, such as `Seager2007:H2O`, has no adiabatic gradient, so the integration holds one temperature over it: the temperature at its end nearer the anchor: the surface temperature with the surface anchor, the mantle temperature at the boundary with the CMB anchor.

**Where this is checked.** [The adiabat across the mantle and ice boundary](../Validation/thermal_state.md#the-adiabat-across-the-mantle-and-ice-boundary).

## How the density solve reads the core temperature

Which temperature does a core node read while the structure still changes from one outer pass to the next?

An outer pass is one step of the outer loop ([Process flow](process_flow.md#iterative-solution)).
In short: each pass reads a T(P) table built on the pass before.
The table has a mantle part and a core part, split at the CMB of the pass that built it.
A core point reads only the core part, so a mantle point never interpolates between a mantle and a core temperature.

A node, or a point of the density integration, reads the core part when its enclosed mass lies below the core mass of the current pass, and the mantle part otherwise.
This also holds when the CMB of the current pass lies at another pressure than that of the table, which happens on every pass whose structure still moves.
When the core of the pass that built the table holds fewer than 2 nodes of positive pressure, or its mantle fewer than 2 nodes, the table is not split, and every point reads one table of all nodes.

**Where this is checked.** [Temperature at the CMB and the centre](../Validation/thermal_state.md#temperature-at-the-cmb-and-the-centre).

### Pressures outside a part of the table

- Past the high-pressure end of either part, $\log_{10} T$ continues linearly in $\log_{10} P$ with the slope of the end interval of that part, so the centre of a pass with a higher central pressure than the table takes a rising temperature, not the end value of the table.
- Past the low-pressure end of the core part, it continues in the same way.
- Past the low-pressure end of the mantle part, it holds the temperature of that end. At $P \le 0$ (a padded node) the solve takes the surface temperature, or the linear first guess below blend 1.
- A table that is not split behaves as the mantle part at both ends.
- Every lookup returns a temperature from 100 K to $10^5$ K, the maximum of the PALEOS tables.

### Which table the strict test accepts

A table built on a pass far from the planet mass puts its CMB at the wrong pressure.
So the strict accept test (the result key `accepted_on` is `'strict'`) accepts a pass only when the pass that built its table met the mass tolerance of the solve ([Full-adiabat rule](process_flow.md#full-adiabat-rule) gives the tolerance and the other accept tests).

## Known limits of the integration

- **Water below 1 bar.** A water layer, whose table reaches 0.1 Pa, is also isothermal below 1 bar.
- **Layer boundary inside a padded top cell.** The adiabat does not cut a padded cell, and the shell takes one mixture over it ([How the density solve reads the temperature over the top cell](#how-the-density-solve-reads-the-temperature-over-the-top-cell)). When an accepted profile in an adiabatic temperature mode has the mantle and ice boundary in that cell, the solver logs a warning ([the measured size](../Validation/thermal_state.md#the-adiabat-across-the-mantle-and-ice-boundary)).
- **Core at the iron melting curve.** Where the core adiabat meets the melting curve of the PALEOS iron table, the integration follows the $\nabla_\mathrm{ad}$ column into the solid branch, while a core at constant entropy would stay on the melting curve. For 4 solves of `input/default.toml` at 2515 K the integrated centre is 724 to 732 K colder than the path at the constant table entropy of the core top, known to about the 84 to 106 K spacing of the table nodes.

## Ranges and checks of the inputs

This section lists what the routine and the config check refuse, and the check that runs on every result.

- **Anchor.** `anchor` must be `'surface'` or `'cmb'`; any other value raises `ValueError`.
- **Range of the anchors.** `compute_adiabatic_temperature` raises `ValueError` for an anchor temperature outside its range, a bool included:
    - the surface temperature: a real number above 0 K and at most $10^5$ K;
    - with `anchor='cmb'`, the CMB temperature: from 100 K to $10^5$ K.

    `validate_config` refuses the same values before a solve starts ([configuration](../How-to/configuration.md#assumptionsandinitialguesses)). `solver.main` does not call it, and PROTEUS calls `solver.main` with its own configuration dict, so on that path the anchors are checked at the first call of `compute_adiabatic_temperature`.

- **Surface anchor below 100 K.** The routine accepts it, but the solver floors the temperature that the density reads from the adiabat at 100 K, so below 100 K the density uses 100 K.
- **Check.** Each call raises `StructureInvariantError` when $T$ is not finite and positive at every node, rises outward from the anchor, or changes over a layer whose EOS has no temperature dependence.
- **Range of `max_dlnp`.** A real number from $10^{-4}$ to 1; any other value raises `ValueError`.

**Where this is checked.** [Adiabat integration on smooth gradients](../Validation/thermal_state.md#adiabat-integration-on-smooth-gradients), [Adiabat integration on real profiles](../Validation/thermal_state.md#adiabat-integration-on-real-profiles), [Temperature across 1 bar](../Validation/thermal_state.md#temperature-across-1-bar) and [Temperature lookup over the top cell](../Validation/thermal_state.md#temperature-lookup-over-the-top-cell). Unit tests check the anchor ranges and the caps: `tests/test_adiabatic.py::TestMidpointAdiabat::test_anchor_temperature_property_over_its_domain`, `::test_a_huge_nabla_stops_at_the_node_caps`, `::test_a_segment_stops_exactly_at_its_limit_property` and `::test_an_unknown_anchor_raises_value_error` for the routine, and `tests/test_config_validation.py::TestTemperatureValidation::test_anchor_temperature_property_over_its_domain` for the config check.
