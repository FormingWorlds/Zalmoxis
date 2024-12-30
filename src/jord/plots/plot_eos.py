import matplotlib.pyplot as plt
import numpy as np
import os

# Function to read data from a file
def read_eos_data(filename):
    data = np.loadtxt(filename, delimiter=',', skiprows=1)
    pressure = data[:, 1]  # Assuming pressure is in the second column (GPa)
    density = data[:, 0] * 1000  # Assuming density is in the first column (g/cm³), convert to kg/m³
    return pressure, density

# List of filenames
data_files = ['eos_seager07_iron.txt', 'eos_seager07_silicate.txt', 'eos_seager07_water.txt']
data_folder = '../data/'

# Create the plot
plt.figure(figsize=(10, 6))

# Plot the data from each file
for file in data_files:
    filepath = os.path.join(data_folder, file)
    pressure, density = read_eos_data(filepath)
    label = file.split('.')[0].replace('eos_', '').capitalize()
    plt.plot(density, pressure, label=label)

# Set plot labels and title
plt.xlabel('Density (kg/m³)')
plt.ylabel('Pressure (GPa)')
plt.title('Equation of State Data')
plt.legend()

# Set log scale for both axes
plt.xscale('log')
plt.yscale('log')

# Invert the y-axis to make it downward-increasing
plt.gca().invert_yaxis()

# Add dotted horizontal helplines for the pressure in the center of the Earth and at the core-mantle boundary
pressure_center_earth = 364  # Pressure in the center of the Earth in GPa
pressure_cmb = 136  # Pressure at the core-mantle boundary in GPa

plt.axhline(y=pressure_cmb, color='gray', linestyle=':', label="Earth's core-mantle boundary (136 GPa)")
plt.axhline(y=pressure_center_earth, color='gray', linestyle='--', label="Earth's center (364 GPa)")


# Show the plot
plt.legend()
plt.show()