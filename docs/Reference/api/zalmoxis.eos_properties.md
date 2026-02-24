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
| `mantle` | `data/EOS_Seager2007/eos_seager07_silicate.txt` | MgSiO₃ perovskite, Birch–Murnaghan EOS |

---

## Iron / temperature-dependent silicate planets  
**Fe: Seager et al. (2007)**  
**Mantle: Wolf & Bower (2018)**

| Material key | EOS file | Description |
|---|---|---|
| `core` | `data/EOS_Seager2007/eos_seager07_iron.txt` | Iron (Fe), Vinet EOS |
| `solid_mantle` | `data/EOS_WolfBower2018_1TPa/density_solid.dat` | Solid MgSiO₃, RTpress EOS |
| `melted_mantle` | `data/EOS_WolfBower2018_1TPa/density_melt.dat` | Molten MgSiO₃, RTpress EOS |

---

## Water planets  
**Seager et al. (2007), 300 K**

| Material key | EOS file | Description |
|---|---|---|
| `core` | `data/EOS_Seager2007/eos_seager07_iron.txt` | Iron (Fe), Vinet EOS |
| `mantle` | `data/EOS_Seager2007/eos_seager07_silicate.txt` | MgSiO₃ perovskite |
| `water_ice_layer` | `data/EOS_Seager2007/eos_seager07_water.txt` | Water ice (phases VIII / X) |

---
