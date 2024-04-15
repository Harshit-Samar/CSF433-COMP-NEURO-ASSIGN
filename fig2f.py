import numpy as np
import matplotlib.pyplot as plt

# Define parameters
sigma_mua = 0.19  # Spatial spread parameter for MUA

# Define range and step for visual angle
y_range = np.arange(-2, 2, 0.01)  # Adjusted range from -2 to 2 degrees

# Gaussian function for MUA with the specified sigma
def gaussian_mua(x, sigma):
    return np.exp(-(x / sigma)**2 / 2)

# Calculate the Gaussian MUA function
gcMUA = gaussian_mua(y_range, sigma_mua)

# Plot the Gaussian MUA function
plt.figure(figsize=(10, 6))
plt.plot(y_range, gcMUA, label=f'MUA Spatial Spread (σ={sigma_mua}°)')
plt.xlabel('Visual Angle (degrees)')
plt.ylabel('Amplitude')
plt.title('Gaussian Plot of MUA with Spatial Spread')
plt.legend()
plt.grid(True)
plt.show()
