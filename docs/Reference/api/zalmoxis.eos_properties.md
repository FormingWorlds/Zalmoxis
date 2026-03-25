::: zalmoxis.eos_properties
    options:
      members: false
      show_source: true

Material property dictionaries defining which EOS tables are used for each
planetary layer and phase. All EOS are registered in the `EOS_REGISTRY` dict,
keyed by EOS identifier strings (e.g., `"Seager2007:iron"`, `"PALEOS:MgSiO3"`).

All EOS file paths are constructed relative to the environment variable: `ZALMOXIS_ROOT`

---

## EOS Registry

The `EOS_REGISTRY` is a flat dict mapping EOS identifier strings to material property dicts. The `calculate_density()` function uses this registry for dispatch.

### Seager2007 (300 K static)

| EOS identifier | Layer key | EOS file | Description |
|---|---|---|---|
| `Seager2007:iron` | `core` | `data/EOS_Seager2007/eos_seager07_iron.txt` | Fe (epsilon), Vinet EOS fit |
| `Seager2007:MgSiO3` | `mantle` | `data/EOS_Seager2007/eos_seager07_silicate.txt` | MgSiO3 perovskite, 4th-order Birch-Murnaghan EOS |
| `Seager2007:H2O` | `ice_layer` | `data/EOS_Seager2007/eos_seager07_water.txt` | Water ice (VII/VIII/X), experimental + DFT |

---

### WolfBower2018 (T-dependent MgSiO3, up to 1 TPa)

| EOS identifier | Layer key | EOS file | Description |
|---|---|---|---|
| `WolfBower2018:MgSiO3` | `melted_mantle` | `data/EOS_WolfBower2018_1TPa/density_melt.dat` | Molten MgSiO3, RTpress EOS |
| | `solid_mantle` | `data/EOS_WolfBower2018_1TPa/density_solid.dat` | Solid MgSiO3, RTpress EOS |
| | `melted_mantle.adiabat_grad_file` | `data/EOS_WolfBower2018_1TPa/adiabat_temp_grad_melt.dat` | Adiabatic gradient (dT/dP)_S for melt |

---

### RTPress100TPa (extended melt to 100 TPa)

| EOS identifier | Layer key | EOS file | Description |
|---|---|---|---|
| `RTPress100TPa:MgSiO3` | `melted_mantle` | `data/EOS_RTPress_melt_100TPa/density_melt.dat` | Molten MgSiO3, extended RTpress (P: 1e3-1e14 Pa) |
| | `solid_mantle` | `data/EOS_WolfBower2018_1TPa/density_solid.dat` | Solid MgSiO3, clamped at 1 TPa |
| | `melted_mantle.adiabat_grad_file` | `data/EOS_RTPress_melt_100TPa/adiabat_temp_grad_melt.dat` | Adiabatic gradient for melt |

---

### PALEOS-2phase (separate solid/liquid MgSiO3)

| EOS identifier | Layer key | EOS file | Description |
|---|---|---|---|
| `PALEOS-2phase:MgSiO3` | `melted_mantle` | `data/EOS_PALEOS_MgSiO3/paleos_mgsio3_tables_pt_proteus_liquid.dat` | Liquid MgSiO3 with nabla_ad (format: `paleos`) |
| | `solid_mantle` | `data/EOS_PALEOS_MgSiO3/paleos_mgsio3_tables_pt_proteus_solid.dat` | Solid MgSiO3 with nabla_ad (format: `paleos`) |

---

### Unified PALEOS (single file per material, all phases)

| EOS identifier | EOS file | Description |
|---|---|---|
| `PALEOS:iron` | `data/EOS_PALEOS_iron/paleos_iron_eos_table_pt.dat` | Fe, 5 phases (alpha-bcc, delta-bcc, gamma-fcc, epsilon-hcp, liquid). Format: `paleos_unified`. |
| `PALEOS:MgSiO3` | `data/EOS_PALEOS_MgSiO3_unified/paleos_mgsio3_eos_table_pt.dat` | MgSiO3, 6 phases (3 pyroxene, bridgmanite, postperovskite, liquid). Format: `paleos_unified`. |
| `PALEOS:H2O` | `data/EOS_PALEOS_H2O/paleos_water_eos_table_pt.dat` | H2O, 7 EOS (ice Ih-X, liquid, vapor, superionic). Format: `paleos_unified`. |

Unified tables derive their phase boundary from the `phase` column at load time. No external melting curves needed. The `mushy_zone_factor` config parameter controls an optional artificial mushy zone below the extracted liquidus.

---

### Chabrier+2019/2021 (pure H2, T-dependent)

| EOS identifier | EOS file | Description |
|---|---|---|
| `Chabrier:H` | `data/EOS_Chabrier2021_HHe/chabrier2021_H.dat` | Pure H$_2$ (molecular, atomic, ionized). DirEOS2021 table from [Chabrier et al. (2019)](https://doi.org/10.3847/1538-4357/aaf99f) / [Chabrier & Debras (2021)](https://doi.org/10.3847/1538-4357/abfc48). Grid: 121 $\times$ 441 ($\log T$, $\log P$), $T = 100$ to $10^8$ K, $P = 1$ Pa to $10^{22}$ Pa. Format: `paleos_unified`. Loaded through the same reader as unified PALEOS tables. |

Additional tables are available in the same data directory (`chabrier2021_HE.dat`, `chabrier2021_HHe_Y0275.dat`, etc.) but are not registered in the EOS registry. Only pure H$_2$ is currently used.
