import numpy as np
import matplotlib.pyplot as plt

# Parameters
num_steps = 2000
window = 50  # Window size for path dependence
attractor_strength = 0.03  # Gentle pull toward moving average
noise_scale = 0.008  # Lower noise for smoother path

x = [0]
y = [0]

for i in range(1, num_steps):
    # Calculate moving average of recent positions
    start = max(0, i - window)
    mean_x = np.mean(x[start:i])
    mean_y = np.mean(y[start:i])
    
    # Small random increments with gentle drift
    dx = noise_scale * np.random.randn() + 0.0005
    dy = noise_scale * np.random.randn() - 0.0005
    
    # Soft path dependence
    dx += attractor_strength * (mean_x - x[-1])
    dy += attractor_strength * (mean_y - y[-1])
    
    x.append(x[-1] + dx)
    y.append(y[-1] + dy)

plt.figure(figsize=(8, 8))
plt.plot(x, y, color='darkred', linewidth=2)
plt.axis('off')
plt.tight_layout()
plt.show()
