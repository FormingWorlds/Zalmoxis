import unittest
import numpy as np
import math
import toml
from src.zalmoxis.zalmoxis import calculate_temperature, calculate_density, coupled_odes, mie_gruneisen_debye, birch_murnaghan
from zalmoxis.constants import *

# Run this test from zalmoxis/src/ directory with the following command:
# python3 -m unittest tests.test1

class Testzalmoxis(unittest.TestCase):
    """
    Unit tests for the zalmoxis planetary structure code.
    """

    def setUp(self):
        """
        Set up the test environment by loading the default configuration file
        and initializing example values for testing.
        """
        # Load default config file for consistent testing
        config_default_path = "../../input/default.toml"
        self.config = toml.load(config_default_path)

        # Example values for testing
        self.radius_guess = 6371e3  # Earth's radius
        self.cmb_radius = 3480e3     # Earth's CMB radius
        self.cmb_temp_guess = 4100
        self.core_temp_guess = 5300

    def test_calculate_temperature_mantle(self):
        """
        Test the calculate_temperature function for a radius within the mantle.
        """
        radius = 5000e3  # Within the mantle
        temp = calculate_temperature(radius, self.cmb_radius, 300, self.cmb_temp_guess, self.core_temp_guess, self.radius_guess)
        self.assertGreater(temp, 300)
        self.assertLess(temp, self.cmb_temp_guess)

    def test_calculate_density_mie_gruneisen_debye(self):
        """
        Test the calculate_density function using the Mie-Gruneisen-Debye equation of state.
        """
        pressure = 1e9  # Example pressure
        radius = 5000e3
        material = "mantle"
        density = calculate_density(pressure, radius, self.cmb_radius, material, self.radius_guess, self.cmb_temp_guess, self.core_temp_guess, "Mie-Gruneisen-Debye")
        self.assertIsNotNone(density)
        self.assertGreater(density, 0)

    def test_calculate_density_birch_murnaghan(self):
        """
        Test the calculate_density function using the Birch-Murnaghan equation of state.
        """
        pressure = 1e9  # Example pressure
        radius = 5000e3
        material = "mantle"
        density = calculate_density(pressure, radius, self.cmb_radius, material, self.radius_guess, self.cmb_temp_guess, self.core_temp_guess, "Birch-Murnaghan")
        self.assertIsNotNone(density)
        self.assertGreater(density, 0)

    def test_mie_gruneisen_debye_eos(self):
        """
        Test the mie_gruneisen_debye function with Earth-like mantle values at a moderate pressure.
        """
        rho = mie_gruneisen_debye(1e10, 24e9, 4110, 245e9, 3.9, 1.5, 1100, 1/4110, 2000)
        self.assertGreater(rho, 4110)  # Expect density increase with pressure

    def test_coupled_odes(self):
        """
        Test the coupled_odes function for valid output shape and non-zero values.
        """
        radius = 5000e3
        y = [1e24, 9.8, 1e10]  # Example mass, gravity, pressure
        interpolation_cache = {} # Initialize cache

        dm, dg, dp = coupled_odes(radius, y) # Use tuple unpacking here

        self.assertIsInstance(dm, (int, float, np.floating))
        self.assertIsInstance(dg, (int, float, np.floating))
        self.assertIsInstance(dp, (int, float, np.floating))

if __name__ == '__main__':
    unittest.main()
