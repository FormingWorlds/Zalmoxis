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
TDEP_EOS_NAMES = {'WolfBower2018:MgSiO3', 'RTPress100TPa:MgSiO3'}

# Other constants
G = 6.67428e-11  # Gravitational constant (m^3 kg^-1 s^-2)
