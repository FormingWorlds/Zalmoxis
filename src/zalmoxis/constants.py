# Reference values for Earth
from __future__ import annotations

earth_radius = (
    6.335439e6  # m — volumetric mean radius used in Seager+2007 (differs from IAU 6.371e6 m)
)
earth_cmb_radius = 3480e3  # m
earth_surface_pressure = 101325  # Pa (1 atm)
earth_cmb_pressure = 135e9  # Pa
earth_center_pressure = 365e9  # Pa (Nimmo 2015)
earth_surface_temperature = 288  # K
earth_cmb_temperature = 4100  # K
earth_center_temperature = 5300  # K
earth_center_density = 13000  # kg/m^3
earth_mass = 5.972e24  # kg


# Temperature-dependent EOS identifiers.  Used throughout the code to decide
# whether temperature profiles, melting curves, and split-radial integration
# are needed.
TDEP_EOS_NAMES = {
    'WolfBower2018:MgSiO3',
    'RTPress100TPa:MgSiO3',
    'PALEOS-2phase:MgSiO3',
    'PALEOS:iron',
    'PALEOS:MgSiO3',
    'PALEOS:H2O',
}

# Phase-aware mixing: smooth density suppression defaults.
# Components with density below the sigmoid center are progressively
# excluded from the harmonic-mean density, preventing non-condensed
# volatiles (vapor, supercritical gas) from dominating the mixture.
CONDENSED_RHO_MIN_DEFAULT = (
    300.0  # kg/m^3, sigmoid center (near H2O critical density 322 kg/m^3)
)
CONDENSED_RHO_SCALE_DEFAULT = 50.0  # kg/m^3, sigmoid transition width

# Other constants
G = 6.67428e-11  # Gravitational constant (m^3 kg^-1 s^-2)
