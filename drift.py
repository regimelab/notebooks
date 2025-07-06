import numpy as np
import matplotlib.pyplot as plt

# Parameters
num_steps = 2000
np.random.seed(42)

# Random walk increments (small random steps + drift)
dx = 0.02 * np.random.randn(num_steps) + 0.001  # small drift in x
dy = 0.02 * np.random.randn(num_steps) - 0.001  # small drift in y

# Cumulative sum to get the path
x = np.cumsum(dx)
y = np.cumsum(dy)

plt.figure(figsize=(8, 8))
plt.plot(x, y, color='darkred', linewidth=2)
plt.axis('off')
plt.tight_layout()
plt.show()
