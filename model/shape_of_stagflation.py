import numpy as np
import matplotlib.pyplot as plt
from polygon import RESTClient
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

API_KEY = "RUDQOgnPw23lKPF5P5y5E_SWufph9tpu"
TICKERS = ["GLD", "USO", "TLT", "SPY"]

def fetch_hourly_returns(ticker, lookback_days=5):
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
    returns = np.diff(np.log(closes))  # log returns
    return returns

# Fetch returns for all tickers
all_returns = []
for tkr in TICKERS:
    ret = fetch_hourly_returns(tkr)
    all_returns.append(ret)

# Align return lengths by truncation to minimum length
min_len = min(len(r) for r in all_returns)
all_returns = [r[-min_len:] for r in all_returns]

# Create returns matrix: shape (time, assets)
returns_matrix = np.array(all_returns).T

# Standardize returns before PCA
scaler = StandardScaler()
returns_scaled = scaler.fit_transform(returns_matrix)

# Fit PCA - keep all components
pca = PCA(n_components=len(TICKERS))
pca.fit(returns_scaled)
components = pca.components_  # shape (n_components, n_features)
explained_var = pca.explained_variance_ratio_

# Visualize top 3 eigenvectors in 3D (first 3 PCA components)
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

x = components[0]
y = components[1] if len(components) > 1 else np.zeros_like(x)
z = components[2] if len(components) > 2 else np.zeros_like(x)


for i in range(len(TICKERS)):
    ax.scatter(x[i], y[i], z[i], label=TICKERS[i])

ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")
ax.set_title("Top 3 PCA Eigenvector Coefficients by Asset")
ax.legend()
plt.show()

# Normalize PCA components for long-only portfolios
components_nonnegative = np.where(components > 0, components, 0)
components_normalized = components_nonnegative / components_nonnegative.sum(axis=1, keepdims=True)
print(components_normalized)

# Calculate portfolio returns by weighting original returns
portfolio_returns = returns_matrix @ components_normalized.T

# Calculate cumulative return and exponential growth for portfolios
cum_returns = np.cumsum(portfolio_returns, axis=0)
portfolio_growth = np.exp(cum_returns) - 1

# Plot portfolio performance for top 3 eigenportfolios
plt.figure(figsize=(12, 6))
for i in range(min(3, len(TICKERS))):
    plt.plot(portfolio_growth[:, i], label=f"Eigenportfolio {i+1} (Explained Var: {explained_var[i]:.2f})")

plt.title("Cumulative Returns of Positive-normalized Eigenportfolio-weighted Portfolios")
plt.xlabel("Time (Hourly)")
plt.ylabel("Cumulative Return")
plt.legend()
plt.show()

