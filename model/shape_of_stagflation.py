import numpy as np
import matplotlib.pyplot as plt
from polygon import RESTClient
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

API_KEY = ""
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

# Fetch returns
all_returns = []
for tkr in TICKERS:
    ret = fetch_hourly_returns(tkr)
    all_returns.append(ret)

min_len = min(len(r) for r in all_returns)
all_returns = [r[-min_len:] for r in all_returns]

returns_matrix = np.array(all_returns).T

# Standardize returns and fit PCA
scaler = StandardScaler()
returns_scaled = scaler.fit_transform(returns_matrix)
pca = PCA(n_components=len(TICKERS))
pca.fit(returns_scaled)
components = pca.components_
explained_var = pca.explained_variance_ratio_

# Long-only normalization of PCA weights
components_nonnegative = np.where(components > 0, components, 0)
components_normalized = components_nonnegative / components_nonnegative.sum(axis=1, keepdims=True)

np.set_printoptions(precision=3, suppress=True)
print("Top 3 Eigenportfolio Weights (Long-Only, normalized):")
for i in range(min(3, len(TICKERS))):
    print(f"Eigenportfolio {i+1}: {components_normalized[i]}")

# 3D Arrow visualization using quiver
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Origins of arrows (all zeros)
origin = np.zeros(len(TICKERS))

# Plot the top 3 eigenportfolio vectors as arrows in 3D
colors = ['r', 'g', 'b']
for i in range(min(3, len(components))):
    vec = components[i]
    ax.quiver(0, 0, 0, vec[0], vec[1] if len(vec) > 1 else 0, vec[2] if len(vec) > 2 else 0,
              color=colors[i], length=1, normalize=True, label=f"Eigenportfolio {i+1}")

ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([-1, 1])
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")
ax.set_title("Top 3 PCA Eigenportfolios as Arrows in Asset Space")
ax.legend()
plt.show()

# Portfolio returns and plotting
portfolio_returns = returns_matrix @ components_normalized.T
cum_returns = np.cumsum(portfolio_returns, axis=0)
portfolio_growth = np.exp(cum_returns) - 1

plt.figure(figsize=(12, 6))
for i in range(min(3, len(TICKERS))):
    plt.plot(portfolio_growth[:, i], color=colors[i], label=f"Eigenportfolio {i+1} (Explained Var: {explained_var[i]:.2f})")

plt.title("Cumulative Returns of Positive-normalized Eigenportfolio-weighted Portfolios")
plt.xlabel("Time (Hourly)")
plt.ylabel("Cumulative Return")
plt.legend()
plt.show()

