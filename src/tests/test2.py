import unittest
import numpy as np
import matplotlib.pyplot as plt
import toml
from jord.jord import main
from jord.constants import *

# Run this test from Jord/src/ directory with the following command:
# python3 -m unittest tests.test2 

class TestJordVaryingMass(unittest.TestCase):

    def setUp(self):
        """Set up for the test by loading the default config."""
        config_default_path = "../../input/default.toml"  # Path adjusted for test location
        self.config = toml.load(config_default_path)

    def test_varying_planet_mass(self):
        """Test Jord with different planet masses and plot the results."""

        masses = [1, 2, 3]  # Earth masses
        results = {}

        for mass in masses:
            # Modify the config for the current mass
            self.config["planet"]["mass"] = mass * earth_mass

            # Run the main code and store results
            radius, density, gravity, pressure = main(self.config)
            results[mass] = (radius, density, gravity, pressure)

        # Plotting the results
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))  # 1 row, 3 columns of plots

        for mass, (radius, density, gravity, pressure) in results.items():
            axes[0].plot(radius / 1000, density / 1000, label=f"{mass} M_earth")
            axes[1].plot(radius / 1000, pressure / 1e9, label=f"{mass} M_earth")
            axes[2].plot(radius / 1000, gravity, label=f"{mass} M_earth")

        axes[0].set_xlabel("Radius (km)")
        axes[0].set_ylabel("Density (g/cm^3)")
        axes[0].set_title("Density Profile")
        axes[0].legend()

        axes[1].set_xlabel("Radius (km)")
        axes[1].set_ylabel("Pressure (GPa)")
        axes[1].set_title("Pressure Profile")
        axes[1].legend()

        axes[2].set_xlabel("Radius (km)")
        axes[2].set_ylabel("Gravity (m/s^2)")
        axes[2].set_title("Gravity Profile")
        axes[2].legend()


        plt.tight_layout()  # Ensures labels don't overlap
        plt.savefig("varying_mass_profiles.png") # Save to a file for visual inspection


if __name__ == '__main__':
    unittest.main()

