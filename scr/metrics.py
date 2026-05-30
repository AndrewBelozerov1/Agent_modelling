import numpy as np
from nolds import hurst_rs, lyap_r, sampen

def calculate_hurst_exponent(price_series):
    if len(price_series) < 30:
        return np.nan
    log_returns = np.diff(np.log(price_series))
    log_returns = log_returns[~np.isnan(log_returns)]
    if len(log_returns) < 10:
        return np.nan
    try:
        hurst = hurst_rs(log_returns)
        return hurst
    except:
        return np.nan

def calculate_lyapunov_exponent(price_series, emb_dim=5, lag=1):
    if len(price_series) < 50:
        return np.nan
    log_returns = np.diff(np.log(price_series))
    log_returns = log_returns[~np.isnan(log_returns)]
    if len(log_returns) < 30:
        return np.nan
    try:
        lyap = lyap_r(log_returns, emb_dim=emb_dim, lag=lag)
        return lyap
    except:
        return np.nan

def calculate_kolmogorov_entropy(price_series, emb_dim=3):
    if len(price_series) < 100:
        return np.nan
    log_returns = np.diff(np.log(price_series))
    log_returns = log_returns[~np.isnan(log_returns)]
    if len(log_returns) < 50:
        return np.nan
    tolerance = 0.2 * np.std(log_returns)
    if tolerance == 0:
        tolerance = 0.01
    try:
        entropy = sampen(log_returns, emb_dim=emb_dim, tolerance=tolerance)
        return entropy
    except:
        return np.nan
