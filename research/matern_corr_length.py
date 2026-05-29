import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from polygon import RESTClient
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as C


# =========================
# CONFIG
# =========================
API_KEY = ""
TICKER = "SPY"
FROM_DATE = "2026-05-10"
TO_DATE   = "2026-05-20"
OUTDIR = "output"
os.makedirs(OUTDIR, exist_ok=True)
client = RESTClient(API_KEY)


# =========================
# DATA PREP (Intraday Only)
# =========================
def fetch_and_clean_hourly(ticker, from_date, to_date):
    bars = [a for a in client.list_aggs(ticker, 1, "minute", from_date, to_date, limit=50000)]
    s = pd.Series([b.close for b in bars], index=pd.to_datetime([b.timestamp for b in bars], unit="ms"))
    s = s[(s.index.hour >= 9) & (s.index.hour <= 16)]  
    # market hours only
    return s.sort_index()

close = fetch_and_clean_hourly(TICKER, FROM_DATE, TO_DATE)
df = pd.DataFrame({"close": close})

# Within‑day squared log returns (overnight gaps removed)
df["log_ret"] = np.log(df["close"]).groupby(df.index.date).diff()**2
df = df.dropna().copy()
df["t"] = np.arange(len(df), dtype=float)
y = df["log_ret"].values                        # raw squared log returns (for GP)
y_std = (y - y.mean()) / y.std()                # standardized volatility
X_t = df["t"].values.reshape(-1, 1)
p = df["close"].values            # raw price (right axis)

# =========================
# DATA‑DRIVEN AXES LIMITS
# =========================
WINDOW = 60
overlap = 1
NLACS = WINDOW/2  # number of lags for ACF

# Compute y ranges from actual data
y_std_min, y_std_max = y_std.min(), y_std.max()
p_min, p_max = p.min(), p.max()
y_std_pad = 0.5 * (y_std_max - y_std_min) if y_std_min != y_std_max else 1.0
p_pad = 0.5 * (p_max - p_min) if p_min != p_max else 1.0
y_std_lims = (y_std_min - y_std_pad, y_std_max + y_std_pad)
p_lims = (p_min - p_pad,        p_max + p_pad)

# We'll store the learned length scale as we go
ls_list = []


# =========================
# GP KERNEL: let length_scale change
# =========================
kernel = (
    C(1.0, (1e-2, 1e3))
    * Matern(length_scale=2.0, nu=1.5, length_scale_bounds=(0.1, 10.0))  # now optimized
    + WhiteKernel(0.01, (1e-4, 0.1))
)


# =========================
# ANIMATION FIGURE: dual y‑axis top + ACF + length_scale panel
# =========================

fig, (ax_top, ax_acf, ax_ls) = plt.subplots(
    3, 1, figsize=(12, 10), sharex=False, gridspec_kw=dict(height_ratios=[2, 1, 1])
)

# TOP: left axis: standardized volatility + GP fit
line_data, = ax_top.plot([], [], 'ko', ms=3, alpha=0.4, label='Standardized volatility')
line_gp,  = ax_top.plot([], [], 'g-', lw=2, label='GP Mean')
fill_gp = ax_top.fill_between([], [], [], color='g', alpha=0.2)

ax_top.set_ylabel("Standardized volatility (fitted GP)")
ax_top.legend(loc='upper left')
ax_top.set_xlim(0, WINDOW - 1)
ax_top.set_ylim(*y_std_lims)

# TOP: right axis: raw price as a line
ax_price = ax_top.twinx()
line_price, = ax_price.plot([], [], lw=1.5, alpha=0.8, label='Raw price')
ax_price.set_ylabel("Price", color='black')
ax_price.tick_params(axis='y', labelcolor='black')
ax_price.set_ylim(*p_lims)


# BOTTOM 1: rolling ACF (standardized volatility)
ax_acf.set_xlabel("Lag")
ax_acf.set_ylabel("Autocorrelation")
ax_acf.set_xlim(0, NLACS)
ax_acf.set_ylim(-0.5, 1.0)

acf_line_0 = ax_acf.axhline(0, color='k', lw=0.5, ls='--')  # 0‑line
line_acf, = ax_acf.plot([], [], 'C3o-', ms=4, lw=1.5, label='ACF (rolling)')
ax_acf.legend()


# BOTTOM 2: learned length scale over time
ax_ls.set_xlabel("Window start index")
ax_ls.set_ylabel("Learned length scale")
ax_ls.set_ylim(0.1, 10.0)  # matches length_scale_bounds
line_ls, = ax_ls.plot([], [], 'C0o-', ms=4, lw=1.5, label='Length scale')
ax_ls.legend()


# =========================
# ANIMATION: rolling window + rolling ACF + learned length scale
# =========================

def animate(i):
    global fill_gp, ls_list

    #import time
    #time.sleep(1)
    start = i
    end = start + WINDOW

    if end > len(df):
        return line_data, line_gp, fill_gp, line_price, line_acf, line_ls

    X_win = X_t[start:end] - X_t[start]  # local time index
    y_win_std = y_std[start:end]

    # Fit GP on standardized volatility (now length_scale is learned)
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True).fit(X_win, y_win_std)
    x_pred = np.linspace(0, WINDOW-1, 80).reshape(-1, 1)
    ym, ys = gp.predict(x_pred, return_std=True)

    # 1. TOP left axis: standardized volatility + GP
    line_data.set_data(np.arange(WINDOW), y_win_std)
    line_gp.set_data(x_pred, ym)
    fill_gp.remove()
    fill_gp = ax_top.fill_between(
        x_pred[:, 0],
        ym - 1.96*ys,
        ym + 1.96*ys,
        color='g', alpha=0.2
    )

    # 2. TOP right axis: raw price lines
    line_price.set_data(np.arange(WINDOW), p[start:end])

    # 3. BOTTOM 1: rolling ACF (lags 0..NLACS)
    from statsmodels.tsa.stattools import acf
    acf_vals = acf(y_win_std, nlags=NLACS, fft=True, missing='conservative')
    lags = np.arange(len(acf_vals))
    line_acf.set_data(lags, acf_vals)

    # 4. BOTTOM 2: learned length scale for this window
    learned_ls = float(gp.kernel_.k1.k2.length_scale)  # C * Matern; the Matern component
    ls_list.append(learned_ls)
    windows_tracked = np.array(ls_list)
    idx_ls = np.arange(len(ls_list))
    line_ls.set_data(idx_ls, windows_tracked)

    # Optionally keep ls axis readable
    if len(ls_list) < 10:
        ax_ls.set_xlim(0, 10)
    else:
        ax_ls.set_xlim(0, len(ls_list))

    return line_data, line_gp, fill_gp, line_price, line_acf, line_ls


frames = np.arange(0, len(df) - WINDOW + 1, overlap)

ani = FuncAnimation(
    fig, animate, frames=frames,
    init_func=lambda: (line_data, line_gp, fill_gp, line_price, line_acf, line_ls),
    blit=False, interval=200, repeat=False
)

plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "spy_rolling_gp_with_price_acf_ls.png"))
plt.show()

# Optional: save animation as GIF
writer = PillowWriter(fps=5)
ani.save(os.path.join(OUTDIR, "spy_rolling_gp_with_price_acf_ls.gif"), writer=writer)
