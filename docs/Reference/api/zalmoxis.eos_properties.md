::: zalmoxis.eos_properties
    options:
      members: false
      show_source: true

Material property dictionaries defining which EOS tables are used for each
planetary layer and phase.

All EOS file paths are constructed relative to the environment variable: `ZALMOXIS_ROOT`

---

## Iron / silicate planets
**Seager et al. (2007), 300 K**

| Material key | EOS file | Description |
|---|---|---|
| `core` | `data/EOS_Seager2007/eos_seager07_iron.txt` | Iron (Fe), Vinet EOS fit |
| `mantle` | `data/EOS_Seager2007/eos_seager07_silicate.txt` | MgSiO₃ perovskite, fourth-order Birch–Murnaghan EOS |

---

## Iron / temperature-dependent silicate planets
**Fe: Seager et al. (2007)**  
**Mantle: Wolf & Bower (2018)**

| Material key | EOS file | Description |
|---|---|---|
| `core` | `data/EOS_Seager2007/eos_seager07_iron.txt` | Iron (Fe), Vinet EOS |
| `melted_mantle` | `data/EOS_WolfBower2018_1TPa/density_melt.dat` | Molten MgSiO₃, RTpress EOS |
| `solid_mantle` | `data/EOS_WolfBower2018_1TPa/density_solid.dat` | Solid MgSiO₃, RTpress EOS |

---

## Iron / RTPress 100 TPa silicate planets
**Fe: Seager et al. (2007)**  
**Melt: extended RTpress table (to 100 TPa)**  
**Solid: Wolf & Bower (2018) / Mosenfelder et al. (2009) (clamped at 1 TPa boundary)**

| Material key | EOS file | Description |
|---|---|---|
| `core` | `data/EOS_Seager2007/eos_seager07_iron.txt` | Iron (Fe), Vinet EOS |
| `melted_mantle` | `data/EOS_RTPress_melt_100TPa/density_melt.dat` | Molten MgSiO₃, extended RTpress EOS table (P: 1e3–1e14 Pa, T: 400–50000 K) |
| `solid_mantle` | `data/EOS_WolfBower2018_1TPa/density_solid.dat` | Solid MgSiO₃, from Wolf & Bower (2018) / Mosenfelder et al. (2009), clamped at 1 TPa |

---

## Water planets
**Seager et al. (2007), 300 K**

| Material key | EOS file | Description |
|---|---|---|
| `core` | `data/EOS_Seager2007/eos_seager07_iron.txt` | Iron (Fe), Vinet EOS |
| `mantle` | `data/EOS_Seager2007/eos_seager07_silicate.txt` | MgSiO₃ perovskite, Birch–Murnaghan EOS |
| `ice_layer` | `data/EOS_Seager2007/eos_seager07_water.txt` | Water ice (phases VIII / X), experimental + DFT-based EOS |