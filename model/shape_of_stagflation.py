import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from polygon import RESTClient
from sklearn.decomposition import PCA
from scipy.stats import zscore
import seaborn as sns
import os
import joblib
from sklearn.mixture import BayesianGaussianMixture

API_KEY = ""
TICKERS = ["SPY", "USO", "GLD", "TLT"]


def eigenports(rets, num_components=3):
  '''
  reduce dimensionality via PCA
  generate orthonormal investment Universes (principal Eigenportfolios)
  
  rets - returns matrix
  num_components - how many PCA vectors to discover
  '''
  eigenvecs = []    
  cols = rets.columns.unique()  
  feat_num = len(cols)

  pca = PCA(n_components=num_components)  
  components = pca.fit_transform(rets[cols].cov())

  for M in range(num_components):
    eigenvec = pd.Series(index=range(feat_num), data=pca.components_[M])
    eigenvecs.append(eigenvec)

  return eigenvecs, pca, components

def fetch_hourly_returns(ticker, lookback_days=18):
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
    closes = np.array([bar.close for bar in bars])
    returns = np.sort(np.diff(np.log(closes)))
    return returns

# Fetch returns for all tickers and align lengths
all_returns = [fetch_hourly_returns(tkr) for tkr in TICKERS]
min_len = min(len(r) for r in all_returns)
all_returns = [r[-min_len:] for r in all_returns]

# Plot raw returns for inspection
for i, ret in enumerate(all_returns):
    sns.lineplot(data=ret, label=f'Returns {TICKERS[i]}')
plt.legend()
plt.show()

# Get eigenportfolios
returns_matrix = pd.DataFrame(all_returns).transpose()
eigenvectors, pca, components = eigenports(returns_matrix)

# Normalize eigenportfolios to long only portfolios:
# Take absolute values and normalize to sum to one per eigenvector 
eigenvectors_long_only = np.abs(eigenvectors)
eigenvectors_long_only /= eigenvectors_long_only.sum(axis=1, keepdims=True)

np.set_printoptions(precision=3, suppress=True)
print("Top 3 Eigenportfolio Weights (Long-Only Normalized):")
for i in range(min(3, len(TICKERS))):
    print(f"Eigenportfolio {i+1}: {eigenvectors_long_only[i]}")

# 3D arrow visualization of top 3 eigenportfolios
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
colors = ['r', 'g', 'b']
for i in range(min(3, len(components))):
    vec = components[i]
    ax.quiver(0, 0, 0,
              vec[0],
              vec[1] if len(vec) > 1 else 0,
              vec[2] if len(vec) > 2 else 0,
              color=colors[i % len(colors)],
              length=1, normalize=True, label=f"Eigenportfolio {i+1}")
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([-1, 1])
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")
ax.set_title("Top 3 PCA Eigenportfolios as Arrows in Asset Space")
ax.legend()
plt.show()

# Portfolio returns and plotting cumulative returns
returns_array = returns_matrix.values if hasattr(returns_matrix, "values") else returns_matrix
portfolio_growth = returns_array @ eigenvectors_long_only.T
plt.figure(figsize=(12, 6))
for i in range(min(3, portfolio_growth.shape[1])):
    cum_returns = np.cumsum(portfolio_growth[:, i])
    plt.plot(cum_returns, colors[i % len(colors)],
             label=f"Eigenportfolio {i+1}")

plt.title("Cumulative Returns of Long-Only Normalized Eigenportfolio-weighted Portfolios")
plt.xlabel("Time (Hourly steps)")
plt.ylabel("Cumulative Return")
plt.legend()
plt.grid(True)
plt.show()
