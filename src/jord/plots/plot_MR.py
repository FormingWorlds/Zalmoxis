from ..constants import *
import matplotlib.pyplot as plt
import os

# Run file via command line: python -m src.jord.plots.plot_MR

# Function to plot the mass-radius relationship of planets and compare with Earth-like Rocky (32.5% Fe+67.5% MgSiO3) planets from Zeng et al. (2019)
def plot_mass_radius_relationship(data_file):
    # Set the working directory to the current file
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Read data from Zeng et al. (2019) for Earth-like Rocky (32.5% Fe+67.5% MgSiO3) planets
    zeng_masses = []
    zeng_radii = []

    with open("../../../data/massradiusEarthlikeRockyZeng.txt", 'r') as zeng_file:
        next(zeng_file)  # Skip the header line
        for line in zeng_file:
            mass, radius = map(float, line.split())
            zeng_masses.append(mass)
            zeng_radii.append(radius)

    # Read data from file with calculated planet masses and radii by the model
    masses = []
    radii = []

    with open(data_file, 'r') as file:
        next(file)  # Skip the header line
        for line in file:
            mass, radius = map(float, line.split())
            masses.append(mass/earth_mass)
            radii.append(radius/earth_radius)

    # Plot the MR graph
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(masses, radii, color='red', label='Model Planets')
    ax.plot(zeng_masses, zeng_radii, color='blue', label='Earth-like Rocky Planets (Zeng et al. 2019)')
    ax.set_xlabel('Planet Mass (Earth Masses)')
    ax.set_ylabel('Planet Radius (Earth Radii)')
    ax.set_title('Calculated Mass-Radius Relationship of Planets')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 2)
    ax.legend()
    ax.grid(True)
    plt.savefig("../MR_plot.png")
    #plt.show()

# Example usage
data_file = '../calculated_planet_mass_radius.txt'
plot_mass_radius_relationship(data_file)