
import numpy as np
from scipy.special import kv, gamma
import matplotlib.pyplot as plt

def matern_kernel(tau, length_scale=1.0, nu=0.5):
    """Matérn kernel for path weighting. nu=0.5 is exponential (rough vol)."""
    rho = np.sqrt(2 * nu) * tau / length_scale
    return (2**(1 - nu) / gamma(nu)) * (rho)**nu * kv(nu, rho)

# Simulate log-price path (daily, 252 days)
np.random.seed(42)
dt = 1/252
T = 1.0
times = np.arange(0, T, dt)
n = len(times)
logS = np.cumsum(np.random.normal(0, 0.02, n))  # Rough path
returns = np.diff(logS) / dt  # Instantaneous returns
squared_returns = returns**2

# PDV: vol_t = sqrt( integral kernel(t-s) * squared_returns_s ds )
# Discretized as weighted sum
def pdv_vol(t_idx, kernel=matern_kernel):
    past_tau = times[:t_idx+1][::-1]  # Time lags from t back
    weights = kernel(past_tau)
    weights /= weights.sum()  # Normalize
    hist_vol_sq = squared_returns[:t_idx]
    return np.sqrt(np.average(hist_vol_sq, weights=weights[:-1]))  # Exclude t

# Compute vol path
vols = [pdv_vol(i) for i in range(50, n)]  # Burn-in

# Plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
ax1.plot(times[1:], squared_returns, label='Squared Returns')
ax1t = ax1.twinx()
ax1t.plot(times[50:50+len(vols)], vols, 'r-', label='Matérn PDV', lw=2)
ax1.legend(loc='upper left')
ax1t.legend(loc='upper right')

tau = np.linspace(0, 0.2, 100)
ax2.plot(tau, matern_kernel(tau), label='Matérn Kernel (nu=0.5)')
ax2.set_xlabel('Time Lag'); ax2.legend()
plt.tight_layout()
plt.show()
