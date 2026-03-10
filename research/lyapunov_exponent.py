
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kurtosis

def logistic_map(r, x0, n):
    x = np.zeros(n)
    x[0] = x0
    for i in range(1, n):
        x[i] = r * x[i-1] * (1 - x[i-1])
    return x

# Parameters
r_values = [3.2, 3.8, 4.0]  # Increasing chaos
n = 10000
x0 = 0.5
burn_in = 1000

fig, axs = plt.subplots(2, 1, figsize=(10, 8))

for r in r_values:
    traj = logistic_map(r, x0, n)
    data = traj[burn_in:]  # Discard transients
    
    # Kurtosis
    k = kurtosis(data)
    
    # Approx Lyapunov: average log |df/dx| over orbit
    lambda_est = np.mean(np.log(abs(r * (1 - 2 * traj[burn_in:-1]))))
    
    print(f"r={r}: Lyapunov ≈ {lambda_est:.3f}, Kurtosis={k:.3f}")
    
    axs[0].plot(traj[::10], 'k-', alpha=0.3, label=f"r={r}")
    axs[1].hist(data, bins=50, density=True, alpha=0.5, label=f"r={r}, kurt={k:.1f}")

axs[0].set_title("Trajectories (showing chaos growth)")
axs[0].legend()
axs[1].set_title("Distributions (leptokurtosis with higher λ)")
axs[1].legend()
plt.tight_layout()
plt.savefig('chaos_kurtosis_demo.png')
plt.show()
