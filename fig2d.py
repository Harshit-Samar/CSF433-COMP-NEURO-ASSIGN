import numpy as np
import matplotlib.pyplot as plt

# Define parameters
sigma_lfp = 0.21  # Spatial spread parameter for LFP

# Define range and step for visual angle
y_range = np.arange(-2, 2, 0.01)  # Adjusted range from -2 to 2 degrees

# Gaussian function for LFP with the specified sigma
def gaussian_lfp(x, sigma):
    return np.exp(-(x / sigma)**2 / 2)

# Calculate the Gaussian LFP function
gcLFP = gaussian_lfp(y_range, sigma_lfp)

# Plot the Gaussian LFP function
plt.figure(figsize=(10, 6))
plt.plot(y_range, gcLFP, label=f'LFP Spatial Spread (σ={sigma_lfp}°)')
plt.xlabel('Visual Angle (degrees)')
plt.ylabel('Amplitude')
plt.title('Gaussian Plot of LFP with Spatial Spread')
plt.legend()
plt.grid(True)
plt.show()

