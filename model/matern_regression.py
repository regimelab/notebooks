import pandas as pd
import numpy as np
import requests
from io import StringIO
import certifi
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C, WhiteKernel
import matplotlib.pyplot as plt


verify_cert_path = certifi.where()


def download_fred_data(url, rename_col):
    resp = requests.get(url, verify=verify_cert_path)
    resp.raise_for_status()
    data = StringIO(resp.text)
    df = pd.read_csv(data)
    df.rename(columns={'observation_date': 'Month', rename_col: rename_col}, inplace=True)
    df['Month'] = pd.to_datetime(df['Month'])
    df.set_index('Month', inplace=True)
    df = df.sort_index()
    df.index = df.index.tz_localize(None)
    df.index = df.index.normalize().map(lambda dt: dt.replace(day=1))
    if df.index.duplicated().any():
        df = df[~df.index.duplicated(keep='first')]
    return df


# Download CPI data
cpi_df = download_fred_data('https://fred.stlouisfed.org/graph/fredgraph.csv?id=CPIAUCSL', 'CPIAUCSL')
cpi_df['CPI_yoy_diff'] = cpi_df['CPIAUCSL'].pct_change(periods=12)


# Download Fed balance sheet data
bal_df = download_fred_data('https://fred.stlouisfed.org/graph/fredgraph.csv?id=WALCL', 'WALCL')
bal_df = bal_df.resample('MS').asfreq().ffill()
bal_df.index = bal_df.index.normalize()
bal_df.index = bal_df.index.map(lambda dt: dt.replace(day=1))
if bal_df.index.duplicated().any():
    bal_df = bal_df[~bal_df.index.duplicated(keep='first')]


# Find common dates and filter
common_dates = cpi_df.index.intersection(bal_df.index)
cpi_df = cpi_df.loc[common_dates]
bal_df = bal_df.loc[common_dates]

# CPI data
df = cpi_df[['CPI_yoy_diff']].join(bal_df[['WALCL']])
df = df[~df.index.duplicated(keep='first')]
df.dropna(inplace=True)
y = df['CPI_yoy_diff'].values


# Use two features: time and fed balance sheet difference 
bal_diff = np.diff(df['WALCL'], prepend=df['WALCL'].iloc[0])
bal_diff_scaled = (bal_diff - bal_diff.mean()) / bal_diff.std()
X = np.column_stack([
    np.arange(len(df)),
    bal_diff_scaled
])


scale_factor = np.sqrt(2)


kernel = (
    C(1.0, (1e-3, 1e3)) *
    Matern(length_scale=10.0 * scale_factor,
           length_scale_bounds=(1e-2 * scale_factor, 1e2 * scale_factor),
           nu=3/2) +
    WhiteKernel(noise_level=1, noise_level_bounds=(1e-5, 1e1))
)


gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, normalize_y=True)
gpr.fit(X, y)


# Define samples 
n_samples = 7
last_time_index = X[-1, 0]


# Forecast
X_forecast = np.column_stack([
    last_time_index + np.arange(1, n_samples + 1),
    np.full(n_samples, X[-1, 1])
])


# Single simulation forecast 
y_samples = gpr.sample_y(X_forecast, n_samples=1, random_state=None).flatten()
_, y_std_forecast = gpr.predict(X_forecast, return_std=True)
y_pred, y_std = gpr.predict(X, return_std=True)


all_indices = np.arange(len(df) + n_samples)
all_dates = df.index.append(
    pd.Index([df.index[-1] + pd.offsets.MonthBegin(i + 1) for i in range(n_samples)])
)
date_labels = all_dates.strftime('%Y-%m')


mean_combined = np.concatenate([y_pred, y_samples])
lower_combined = np.concatenate([y_pred - 1.96 * y_std, y_samples - 1.96 * y_std_forecast])
upper_combined = np.concatenate([y_pred + 1.96 * y_std, y_samples + 1.96 * y_std_forecast])
x_combined = np.concatenate([np.arange(len(df)), all_indices[-n_samples:]])


plot_color_observed = 'firebrick'
plot_color_pred = 'black'
plot_color_forecast = 'dimgray'
plot_alpha_forecast = 0.4
plot_marker_observed = 'o'
plot_marker_forecast = 'x'
plot_linewidth = 1.5
plot_markersize = 8


fig, ax1 = plt.subplots(figsize=(12, 6))


ax1.plot(np.arange(len(df)), y, plot_marker_observed, color=plot_color_observed, label='Observed YoY CPI Diff')
ax1.plot(np.arange(len(df)), y_pred, '-', color=plot_color_pred, linewidth=plot_linewidth, label='GP Fit (Historical)')
ax1.plot(x_combined, mean_combined, linestyle='--', color=plot_color_forecast, linewidth=plot_linewidth, label='GP Fit + Sampled Forecast')
ax1.fill_between(x_combined, lower_combined, upper_combined, color=plot_color_forecast, alpha=plot_alpha_forecast)
ax1.plot(all_indices[-n_samples:], y_samples, plot_marker_forecast, color=plot_color_observed, markersize=plot_markersize, label='Sampled Forecast Points')


ax1.set_xlabel('Date')
ax1.set_ylabel('YoY ΔCPI')


# Secondary y-axis for fed balance sheet diff (scaled)
ax2 = ax1.twinx()
ax2.plot(np.arange(len(df)), bal_diff_scaled, color='tab:blue', linestyle='-', alpha=0.7, label='Fed Balance Sheet Diff (scaled)')
ax2.set_ylabel('Fed Balance Sheet Diff (scaled)', color='tab:blue')
ax2.tick_params(axis='y', labelcolor='tab:blue')


lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize='small')


ax1.set_xticks(all_indices[::max(1, len(all_indices) // 10)])
ax1.set_xticklabels(date_labels[::max(1, len(all_indices) // 10)], rotation=45, ha='right')
ax1.set_title(f'GP Fit and {n_samples} Sampled Points Forecast with Secondary Axis for Fed Balance Sheet Diff')
plt.tight_layout()
plt.show()

