import numpy as np
from ..eos_properties import analytical_eos_Seager07_parameters
import matplotlib.pyplot as plt
import os

# Define custom colors
custom_colors = {
    'Fe': 'red',
    'MgSiO3': 'orange',
    'H2O': 'blue',
}

def plot_analytical_eos_all():
    fig, ax = plt.subplots(figsize=(8, 6))  # Create a figure and an axes object
    
    for material in analytical_eos_Seager07_parameters.keys():
        # Set the working directory to the current file
        os.chdir(os.path.dirname(os.path.abspath(__file__))) 

        # Define the file path
        file_path = f"../../../data/analytical_eos_seager07_{material}.txt"
        
        # Load the data
        try:
            data = np.genfromtxt(file_path, delimiter=',', skip_header=1)  # Assuming the first row is a header
            density = data[:, 0]  # First column: Density (g/cm^3)
            pressure = data[:, 1]  # Second column: Pressure (GPa)
        except Exception as e:
            print(f"Error loading data from {file_path}: {e}")
            continue

        # Get the color for the current material
        color = custom_colors.get(material, 'black')  # Default to black if material not in custom_colors

        # Plot the data for the current material
        ax.plot(pressure, density * 1000, label=f'{material} EOS', color=color)  # Convert density to kg/m³
    
    # Set labels, log scale, title, legend, and grid
    ax.set_xlabel('Pressure (GPa)')
    ax.set_ylabel('Density (kg/m³)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title('Analytical EOS for All Materials')
    ax.legend()
    ax.grid(True)
    fig.savefig("planet_analytical_eos.pdf")
    fig.tight_layout()
    #plt.show()

# Example usage
plot_analytical_eos_all()