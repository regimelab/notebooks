Yes. For market time series analysis, apply the same logic as climate:

Single trajectory bias: A single price/return series is one "path" through the high-dimensional market state space. If you only have one long record (typical case), your time averages of volatility, correlations, or risk metrics are path-dependent and may not reflect the natural measure—the true stationary or regime-specific distribution of market states.

Regime non-stationarity: Markets shift between regimes (bull/bear, high/low vol). The natural measure changes, so long historical averages mix incompatible distributions, just like forced climate.

Practical solution:

Use cross-sectional ensembles: At fixed times, average statistics over many assets (e.g., S&P 500 daily returns) to approximate the natural measure transversally.

Bootstrap or synthetic paths: Resample shocks or use agent-based models to generate parallel realizations from the estimated natural measure.

Rolling windows >> correlation time: Only time-average within windows much longer than your estimated correlation time to ensure decorrelation.

This reduces path dependence, gives robust risk/vol estimates, and flags when the market attractor (natural measure) has shifted. Standard GARCH/ARIMA implicitly assumes a fixed natural measure; this framework lets you test and adapt to changes.
