import subprocess
import matplotlib.pyplot as plt
import numpy as np

# First run with default config
subprocess.run(['python', 'jord.py'])
data1 = np.loadtxt('output.txt')

# Subsequent runs with doubled planet mass
subprocess.run(['python', 'jord.py', '--planet_mass', '2'])
data2 = np.loadtxt('output.txt')
subprocess.run(['python', 'jord.py', '--planet_mass', '2'])
data3 = np.loadtxt('output.txt')


# Plotting the results
time = data1[:,0]
plt.plot(time, data1[:,1], label='Default')
plt.plot(time, data2[:,1], label='Doubled Mass')
plt.plot(time, data3[:,1], label='Doubled Mass (Repeat)')
plt.xlabel('Time')
plt.ylabel('Output')
plt.legend()
plt.savefig('comparison.png')
