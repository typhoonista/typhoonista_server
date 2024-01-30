import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
import matplotlib.transforms as transforms

# Define input variables (Soil Moisture Level and Irrigation Control)
soil_moisture_levels = ['Dry', 'Moist', 'Wet']
irrigation_controls = ['Low', 'Medium', 'High']

# Define membership functions
def triangular_mf(x, a, b, c):
    return np.maximum(0, np.minimum((x - a) / (b - a), (c - x) / (c - b)))

def trapezoidal_mf(x, a, b, c, d):
    return np.maximum(0, np.minimum.reduce([(x - a) / (b - a), 1, (d - x) / (d - c)]))

# Plot membership functions
fig, ax = plt.subplots(2, 1, figsize=(8, 8))

# Soil Moisture Level membership functions
ax[0].plot([0, 10, 20], [0, 1, 0], label='Dry')
ax[0].plot([10, 20, 30, 40], [0, 1, 1, 0], label='Moist')
ax[0].plot([30, 40, 50], [0, 1, 0], label='Wet')

# Irrigation Control membership functions
ax[1].plot([0, 10, 20], [0, 1, 0], label='Low')
ax[1].plot([10, 20, 30], [0, 1, 0], label='Medium')
ax[1].plot([20, 30, 40], [0, 1, 0], label='High')

# Add labels and legends
ax[0].set_title('Soil Moisture Level')
ax[0].set_xlabel('Moisture Level')
ax[0].set_ylabel('Membership')
ax[0].legend()

ax[1].set_title('Irrigation Control')
ax[1].set_xlabel('Irrigation Level')
ax[1].set_ylabel('Membership')
ax[1].legend()

# Plot fuzzy rules
for i, soil_mf in enumerate(soil_moisture_levels):
    for j, irrigation_mf in enumerate(irrigation_controls):
        rule_strength = triangular_mf((i + j) * 10, 0, 10, 20)
        color = (0, 0, rule_strength)
        rect = patches.Rectangle((i - 0.5, j - 0.5), 1, 1, linewidth=1, edgecolor='black', facecolor=color)
        ax[0].add_patch(rect)

# Display the plot
plt.show()
