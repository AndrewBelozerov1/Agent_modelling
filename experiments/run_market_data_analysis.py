import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from nolds import hurst_rs, lyap_r, sampen

def calculate_metrics(data, name):
    data['Дата'] = pd.to_datetime(data['Дата'], format='%d.%m.%Y')
    data['Цена'] = data['Цена'].astype(str).str.replace('.', '', regex=False).str.replace(',', '.').astype(float)
    price_series = data['Цена'].values
    log_returns = np.diff(np.log(price_series))
    hurst = hurst_rs(log_returns)
    lyap = lyap_r(log_returns, emb_dim=10, lag=1)
    tolerance = 0.2 * np.std(log_returns)
    if tolerance == 0:
        tolerance = 0.01
    kse = sampen(log_returns, emb_dim=3, tolerance=tolerance)
    print(f"\n{name}")
    print(f"Показатель Херста: {hurst:.4f}")
    print(f"Показатель Ляпунова: {lyap:.6f}")
    print(f"Энтропия Колмогорова-Синая: {kse:.6f} бит/шаг")

if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    
    rts = pd.read_csv(os.path.join(data_dir, 'ртс.csv'))
    mmvb = pd.read_csv(os.path.join(data_dir, 'ммвб.csv'))
    sp500 = pd.read_csv(os.path.join(data_dir, 's&p500.csv'))
    djia = pd.read_csv(os.path.join(data_dir, 'djia.csv'))
    nasdaq = pd.read_csv(os.path.join(data_dir, 'nasdaq.csv'))
    
    calculate_metrics(rts, "РТС")
    calculate_metrics(mmvb, "ММВБ")
    calculate_metrics(sp500, "S&P500")
    calculate_metrics(djia, "DJIA")
    calculate_metrics(nasdaq, "NASDAQ")
