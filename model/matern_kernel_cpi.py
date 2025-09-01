import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import requests
from io import StringIO
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C, WhiteKernel

# Download CPI data CSV from FRED
csv_url = 'https://fred.stlouisfed.org/graph/fredgraph.csv?id=CPIAUCSL'
response = requests.get(csv_url)
response.raise_for_status()
data = StringIO(response.text)
df = pd.read_csv(data)

# Rename columns and parse dates
df.rename(columns={'observation_date': 'Month', 'CPIAUCSL': 'CPI'}, inplace=True)
df['Month'] = pd.to_datetime(df['Month'])
df.set_index('Month', inplace=True)

# Calculate YoY difference and drop NA
df['CPI_yoy_diff'] = df['CPI'].pct_change(periods=12)
diff_series = df['CPI_yoy_diff'].dropna()[:201]

# Prepare data for Gaussian Process regression
X = np.arange(len(diff_series)).reshape(-1, 1)  
y = diff_series.values

import seaborn as sns
fig,ax=plt.subplots()
ax.plot(diff_series.index, y)
plt.suptitle('Inflation rate since 1947')
# Add shaded regions for major events
regions = [
    ('WW2 Debt Payoff', '1947-01', '1955-12', 'lightgray'),
    ('Great Inflation', '1965-01', '1982-12', 'salmon'),
    ('Great Moderation', '1982-01', '2007-12', 'lightblue'),
    ('GFC', '2007-01', '2010-12', 'orange'),
    ('Covid/Covid + Inflation', '2020-01', '2023-12', 'lightgreen')
]
for label, start, end, color in regions:
    ax.axvspan(pd.to_datetime(start), pd.to_datetime(end), color=color, alpha=0.4, label=label)
plt.legend()
plt.show()

# Kernel for GP
kernel = (
    C(1.0, (1e-3, 1e3))
    * Matern(length_scale=10.0, length_scale_bounds=(1e-2, 1e2), nu=1.5)
    + WhiteKernel(noise_level=1, noise_level_bounds=(1e-5, 1e1))
)

gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, normalize_y=True)
gpr.fit(X, y)

# In-sample predictions
X_pred = np.linspace(0, len(diff_series) - 1, len(diff_series)).reshape(-1, 1)
y_pred, y_std = gpr.predict(X_pred, return_std=True)
dates_pred = diff_series.index[0] + pd.offsets.MonthBegin() * X_pred.flatten().astype(int)

# OUT-OF-SAMPLE forward predictions
n_forecast = 1
X_forecast = np.arange(len(diff_series), len(diff_series) + n_forecast).reshape(-1, 1)
y_forecast, y_forecast_std = gpr.predict(X_forecast, return_std=True)
dates_forecast = [diff_series.index[-1] + pd.offsets.MonthBegin(i + 1) for i in range(n_forecast)]

# Combine prediction arrays for plotting
all_dates = np.concatenate([dates_pred, dates_forecast])
all_pred = np.concatenate([y_pred, y_forecast])
all_std = np.concatenate([y_std, y_forecast_std])

# Plot all (observed, GP fit, and forecast with continuous shaded CI)
plt.figure(figsize=(12, 6))
plt.plot(diff_series.index, y, 'r.', label='Observed')
plt.plot(all_dates, all_pred, 'b-', label='GP Matérn fit + Forecast')
plt.fill_between(
    all_dates, all_pred - 1.96 * all_std, all_pred + 1.96 * all_std, alpha=0.2, color='blue',
    label='95% confidence interval'
)
plt.title('GP Matérn Fit to YoY CPI Diff & Forward Forecast')
plt.xlabel('Date')
plt.ylabel('YoY ΔCPI')
plt.legend()
plt.tight_layout()
plt.show()

# Print kernel and forecast values
print("Learned kernel:", gpr.kernel_)
for i in range(n_forecast):
    print(f"Point {i+1} forecast ({dates_forecast[i].strftime('%Y-%m')}): {y_forecast[i]:.4f} ± {y_forecast_std[i]:.4f}")
