import numpy as np
import matplotlib.pyplot as plt
from polygon import RESTClient
from sklearn.decomposition import PCA


API_KEY = ""  
TICKERS = ["GLD", "USO", "TLT", "SPY"]


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


# Fit PCA
pca = PCA(n_components=len(TICKERS))
pca.fit(returns_matrix)
components = pca.components_
explained_var = pca.explained_variance_ratio_


# Long-only normalization of PCA weights
components_nonnegative = np.where(components > 0, components, 0)
components_normalized = components_nonnegative / components_nonnegative.sum(axis=1, keepdims=True)


np.set_printoptions(precision=3, suppress=True)
print("Top 3 Eigenportfolio Weights (Long-Only, normalized):")
for i in range(min(3, len(TICKERS))):
    print(f"Eigenportfolio {i+1}: {components_normalized[i]}")


# 3D Arrow visualization of top 3 eigenportfolios
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

for i in range(min(3, len(components))):
    vec = components[i]
    ax.quiver(0, 0, 0,
              vec[0],
              vec[1] if len(vec) > 1 else 0,
              vec[2] if len(vec) > 2 else 0,
              color=['r', 'g', 'b'][i],
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


# Portfolio returns and plotting
portfolio_returns = returns_matrix @ components_normalized.T
cum_returns = np.cumsum(portfolio_returns, axis=0)
portfolio_growth = np.exp(cum_returns) - 1

plt.figure(figsize=(12, 6))
colors = ['r', 'g', 'b']
for i in range(min(3, len(TICKERS))):
    plt.plot(portfolio_growth[:, i], color=colors[i],
             label=f"Eigenportfolio {i+1} (Explained Var: {explained_var[i]:.2f})")

plt.title("Cumulative Returns of Positive-normalized Eigenportfolio-weighted Portfolios")
plt.xlabel("Time (Hourly)")
plt.ylabel("Cumulative Return")
plt.legend()
plt.show()


# Bar plot with multi-level x-axis labels (Eigenportfolio + Assets)
num_portfolios = min(3, len(TICKERS))
num_assets = len(TICKERS)
bar_width = 0.15

plt.figure(figsize=(12, 6))

# X position for each group (eigenportfolio)
group_positions = np.arange(num_portfolios)

# Plot bars: for each portfolio, plot bars for each asset
for asset_idx in range(num_assets):
    weights = components_normalized[:num_portfolios, asset_idx]
    # Position bars inside group for this asset
    x = group_positions + asset_idx * bar_width
    plt.bar(x, weights, width=bar_width, color="#1f77b4")

# X-axis: place tick labels centered under each eigenportfolio group
center_positions = group_positions + bar_width * (num_assets - 1) / 2
plt.xticks(center_positions, [f"Eigenportfolio {i+1}" for i in range(num_portfolios)], fontsize=12)

# Add secondary x-axis labels for assets below the eigenportfolio labels
ax = plt.gca()
# Coordinates to place asset labels
for i, gp in enumerate(group_positions):
    for j, ticker in enumerate(TICKERS):
        # x position of each bar
        xpos = gp + j * bar_width
        # tiny vertical "tick" line
        ax.plot([xpos, xpos], [-0.02, -0.005], color='black', clip_on=False, transform=ax.get_xaxis_transform())
        # asset label below tick line with smaller font
        ax.text(xpos, -0.05, ticker, ha='center', va='top', fontsize=8, transform=ax.get_xaxis_transform())

plt.title("Weights of Top 3 Eigenportfolios (Long-Only Normalized)")
plt.ylabel("Weight")
plt.ylim(0, components_normalized.max() * 1.1)
plt.tight_layout()
plt.show()

