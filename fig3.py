import numpy as np
import matplotlib.pyplot as plt

# Mean visual spreads for LFP and MUA
mean_LFP = 0.263
mean_MUA = 0.242

# Standard errors for LFP and MUA
se_LFP = 0.062
se_MUA = 0.049

# Generate random visual spreads for LFP and MUA based on means and standard errors
np.random.seed(0)
visual_spread_LFP = np.random.normal(loc=mean_LFP, scale=se_LFP, size=251)
visual_spread_MUA = np.random.normal(loc=mean_MUA, scale=se_MUA, size=251)

# Plotting
plt.figure(figsize=(8, 6))
plt.scatter(visual_spread_LFP, visual_spread_MUA, label='Visual Spreads')
plt.plot([0, 0.5], [0, 0.5], 'k--', label='Unity Line')  # Unity line where LFP visual spread = MUA visual spread
plt.xlabel('Visual Spread for LFP')
plt.ylabel('Visual Spread for MUA')
plt.title('Comparison of Visual Spreads for LFP and MUA')
plt.legend()
plt.grid(True)
plt.xlim(0, 0.5)  # Limit x-axis from 0 to 0.5
plt.ylim(0, 0.5)  # Limit y-axis from 0 to 0.5
plt.show()
