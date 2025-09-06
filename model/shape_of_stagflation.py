import numpy as np
import matplotlib.pyplot as plt
from polygon import RESTClient
from scipy.optimize import minimize
from scipy.linalg import cholesky, cho_solve, kron, LinAlgError

API_KEY = "RUDQOgnPw23lKPF5P5y5E_SWufph9tpu"
TICKERS = ["GLD", "TLT"]

def fetch_hourly_closes(ticker, lookback_days=25):
    client = RESTClient(API_KEY)
    from_date = str((np.datetime64('today') - np.timedelta64(lookback_days, 'D')))
    bars = client.list_aggs(
        ticker=ticker,
        multiplier=1,
        timespan="hour",
        from_=from_date,
        to=str(np.datetime64('today')),
        limit=9999
    )
    closes = [bar.close for bar in bars]
    closes = np.array(closes)
    return np.diff(np.log(closes))  # log returns

# Fetch and prepare data
all_closes = []
for tkr in TICKERS:
    closes = fetch_hourly_closes(tkr)
    all_closes.append(closes)

min_len = min(len(c) for c in all_closes)
all_closes = [c[-min_len:] for c in all_closes]
Y = np.array(all_closes)  # shape: (n_assets, n_samples)

n_assets, n_samples = Y.shape

# Prepare time inputs as 2D array as required by kernel functions
X_time = np.arange(n_samples).reshape(-1, 1)  # shape (n_samples, 1)

def matern_cov(X1, X2, length_scale, nu=0.5):
    # Matern 1/2 kernel (Exponential kernel) manual implementation for optimization
    dists = np.abs(X1 - X2.T)
    return np.exp(-dists / length_scale)

def build_output_cov(params):
    # Parameterize the 2x2 output covariance matrix to be positive semidefinite
    # params = [a, b, c], where diagonal entries are exp to ensure >0
    a, b, c = params
    M = np.array([[np.exp(a), b],
                  [b, np.exp(c)]])
    # We add small jitter later to handle PSD issues if any
    return M

def neg_log_marginal_likelihood(theta):
    length_scale = np.exp(theta[0])  # positive length scale
    output_cov_params = theta[1:]
    
    K_time = matern_cov(X_time, X_time, length_scale)
    K_output = build_output_cov(output_cov_params)
    
    try:
        # Kronecker product covariance matrix
        K = kron(K_output, K_time)
        y_vec = Y.flatten('F')  # flatten in column-major order to match Kronecker layout
        
        # Cholesky decomposition with jitter for numerical stability
        L = cholesky(K + 1e-8 * np.eye(n_assets * n_samples), lower=True)
        alpha = cho_solve((L, True), y_vec)
        
        # Compute negative log marginal likelihood
        nll = 0.5 * y_vec.dot(alpha)
        nll += np.sum(np.log(np.diag(L)))
        nll += 0.5 * n_assets * n_samples * np.log(2 * np.pi)
        return nll
    except LinAlgError:
        # Return large value if covariance not positive definite
        return 1e10

# Initial hyperparameter guess: log length scale and output covariance params
init_theta = np.array([np.log(1.0), np.log(1.0), 0.0, np.log(1.0)])

bounds = [(np.log(1e-2), np.log(1e2)), (None, None), (None, None), (None, None)]

# Optimize hyperparameters to fit the GP to data
result = minimize(neg_log_marginal_likelihood, init_theta, bounds=bounds, method='L-BFGS-B')

opt_theta = result.x
opt_length_scale = np.exp(opt_theta[0])
opt_output_cov = build_output_cov(opt_theta[1:])

print(f"Optimized time length scale: {opt_length_scale}")
print("Optimized output covariance matrix:")
print(opt_output_cov)

# Build covariance matrix with optimized hyperparameters for sampling
K_time_opt = matern_cov(X_time, X_time, opt_length_scale)
K_opt = kron(opt_output_cov, K_time_opt)
K_opt += 1e-8 * np.eye(n_assets * n_samples)  # jitter

# Sample from multivariate normal with optimized covariance
simulated = np.random.multivariate_normal(mean=np.zeros(n_assets * n_samples), cov=K_opt)

# Reshape sampled vector to 2D (n_assets x n_samples)
simulated_paths = simulated.reshape(n_assets, n_samples)

# Plot the simulation results
plt.plot(np.cumsum(simulated_paths[0]), label=TICKERS[0])
plt.plot(np.cumsum(simulated_paths[1]), label=TICKERS[1])
plt.title("Simulated Paths from Hyperparameter-Tuned Kronecker GP")
plt.xlabel("Sample Index")
plt.ylabel("Simulated Value")
plt.legend()
plt.show()

