import matplotlib.pyplot as plt
import numpy as np
import os

# Function to read data from a file
def read_eos_data(filename):
    data = np.loadtxt(filename, delimiter=',', skiprows=1)
    pressure = data[:, 1]
    density = data[:, 0] * 1000  # Convert to kg/m³
    return pressure, density

# List of filenames
data_files = ['eos_iron.txt', 'eos_silicate.txt', 'eos_water.txt']
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

# Show the plot
plt.show()