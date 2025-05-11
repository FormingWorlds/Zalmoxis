import numpy as np
import os

# Set the working directory to the current file
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Read the data from the input file
input_file = "../../../data/raw_eos_chabrier19_HHe.txt"
output_file = "../../../data/eos_chabrier19_HHe.txt"

# Load the data, skipping the header row
data = np.loadtxt(input_file, skiprows=1)

# Extract the 2nd (log rho) and 3rd (log P) columns
log_rho = data[:, 2]
log_P = data[:, 1]

# Convert log values to linear scale
rho = 10**log_rho  # g/cm^3
P = 10**log_P      # GPa

# Save the extracted columns to a new file
np.savetxt(output_file, np.column_stack((rho, P)), header="rho (g/cm^3)    P (GPa)", fmt="%.10e %.6e")