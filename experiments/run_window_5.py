import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import numpy as np
from src.asset import Asset
from src.order_book import OrderBook
from src.traders import NoiseTrader, ChartistTrader, FundamentalistTrader
from src.metrics import calculate_hurst_exponent, calculate_lyapunov_exponent, calculate_kolmogorov_entropy

# Словарь для хранения истории богатства
wealth_history = {
    'days': [],
    'noise_avg': [],
    'chartist_avg': [],
    'fundamentalist_avg': []
}

def run_simulation_window_5(days=1000, noise_traders=50, chartists=20, fundamentalists=30):
    """Симуляция с окном скользящей средней = 5 дней (вместо 20)"""
    asset = Asset()
    order_book = OrderBook()
    traders = []
    
    for _ in range(noise_traders):
        traders.append(NoiseTrader(cash=100000, assets=1000))
    
    # Чартисты с окном 5 дней (очень короткий горизонт)
    for _ in range(chartists):
        traders.append(ChartistTrader(cash=150000, assets=1000, window=5, alpha=0.1))
    
    for _ in range(fundamentalists):
        traders.append(FundamentalistTrader(cash=1000000, assets=500, gamma=2, rf=0.0003))
    
    # Начальный прогрев
    for _ in range(20):
        new_price = asset.current_price * np.exp(0.0003 - 0.02**2/2 + 0.02*np.random.randn())
        asset.update_price(new_price)
        asset.update_dividend()
    
    initial_wealths = {id(trader): trader.wealth_history[0] for trader in traders}
    
    wealth_history['days'] = []
    wealth_history['noise_avg'] = []
    wealth_history['chartist_avg'] = []
    wealth_history['fundamentalist_avg'] = []
    
    for day in range(days):
        asset.update_dividend()
        
        for trader in traders:
            if trader.cash <= 0 and trader.assets <= 0:
                trader.active = False
                continue
            order_type, price, qty = trader.make_decision(asset)
            if order_type and price and qty:
                order_book.add_order(order_type, price, qty, id(trader))
        
        executed = order_book.match_orders(traders, asset)
        
        for trader in traders:
            trader.update_wealth(asset.current_price)
        
        # Логирование каждые 100 дней
        if day % 100 == 0 or day == days-1:
            _log_status(day, asset, traders, noise_traders, chartists, fundamentalists)
    
    # Финальный анализ
    _print_analysis(traders, initial_wealths, asset.price_history, 
                    noise_traders, chartists, fundamentalists, window_size=5)
    
    return asset, traders

def _log_status(day, asset, traders, noise_traders, chartists, fundamentalists):
    noise_wealth, chartist_wealth, fundamentalist_wealth = [], [], []
    noise_stats = {'cash': [], 'assets': [], 'active': 0}
    chart_stats = {'cash': [], 'assets': [], 'active': 0}
    fund_stats = {'cash': [], 'assets': [], 'active': 0}
    
    for trader in traders:
        if not trader.active:
            continue
        current_wealth = trader.cash + trader.assets * asset.current_price
        if trader.type == 'noise':
            noise_wealth.append(current_wealth)
            noise_stats['cash'].append(trader.cash)
            noise_stats['assets'].append(trader.assets)
            noise_stats['active'] += 1
        elif trader.type == 'chartist':
            chartist_wealth.append(current_wealth)
            chart_stats['cash'].append(trader.cash)
            chart_stats['assets'].append(trader.assets)
            chart_stats['active'] += 1
        elif trader.type == 'fundamentalist':
            fundamentalist_wealth.append(current_wealth)
            fund_stats['cash'].append(trader.cash)
            fund_stats['assets'].append(trader.assets)
            fund_stats['active'] += 1
    
    wealth_history['days'].append(day)
    wealth_history['noise_avg'].append(np.mean(noise_wealth) if noise_wealth else 0)
    wealth_history['chartist_avg'].append(np.mean(chartist_wealth) if chartist_wealth else 0)
    wealth_history['fundamentalist_avg'].append(np.mean(fundamentalist_wealth) if fundamentalist_wealth else 0)
    
    print(f"\nDay {day}: Price = {asset.current_price:.2f}")
    print(f"Active Traders - Noise: {noise_stats['active']}/{noise_traders}, "
          f"Chartists (window=5): {chart_stats['active']}/{chartists}, "
          f"Fundamentalists: {fund_stats['active']}/{fundamentalists}")
    
    if noise_stats['active'] > 0:
        print(f"Noise Traders: Avg Wealth = {wealth_history['noise_avg'][-1]:.2f}")
    if chart_stats['active'] > 0:
        print(f"Chartist Traders: Avg Wealth = {wealth_history['chartist_avg'][-1]:.2f}")
    if fund_stats['active'] > 0:
        print(f"Fundamentalist Traders: Avg Wealth = {wealth_history['fundamentalist_avg'][-1]:.2f}")

def _print_analysis(traders, initial_wealths, price_history, noise_traders, chartists, fundamentalists, window_size):
    print(f"АНАЛИЗ РЕЗУЛЬТАТОВ СИМУЛЯЦИИ (окно скользящей средней = {window_size} дней)")
    
    profitable_noise = profitable_chartists = profitable_fundamentalists = 0
    noise_changes, chartist_changes, fundamentalist_changes = [], [], []
    
    for trader in traders:
        if id(trader) not in initial_wealths:
            continue
        initial = initial_wealths[id(trader)]
        final = trader.wealth_history[-1]
        change = ((final - initial) / initial) * 100
        
        if trader.type == 'noise':
            noise_changes.append(change)
            if final > initial:
                profitable_noise += 1
        elif trader.type == 'chartist':
            chartist_changes.append(change)
            if final > initial:
                profitable_chartists += 1
        elif trader.type == 'fundamentalist':
            fundamentalist_changes.append(change)
            if final > initial:
                profitable_fundamentalists += 1
    
    print(f"  Шумовые трейдеры:        {profitable_noise}/{noise_traders} ({profitable_noise/noise_traders*100:.1f}%)")
    print(f"  Технические трейдеры (window={window_size}): {profitable_chartists}/{chartists} ({profitable_chartists/chartists*100:.1f}%)")
    print(f"  Фундаменталисты:         {profitable_fundamentalists}/{fundamentalists} ({profitable_fundamentalists/fundamentalists*100:.1f}%)")
    
    if noise_changes:
        print(f"  Шумовые трейдеры:        {np.mean(noise_changes):+.2f}%")
    if chartist_changes:
        print(f"  Технические трейдеры:     {np.mean(chartist_changes):+.2f}%")
    if fundamentalist_changes:
        print(f"  Фундаменталисты:         {np.mean(fundamentalist_changes):+.2f}%")
    
    # Расчёт метрик
    hurst_val = calculate_hurst_exponent(np.array(price_history))
    lyap_val = calculate_lyapunov_exponent(np.array(price_history), emb_dim=10, lag=1)
    kse_val = calculate_kolmogorov_entropy(np.array(price_history), emb_dim=3)
    
    if not np.isnan(hurst_val):
        print(f"  Показатель Херста (H):    {hurst_val:.4f}")
    if not np.isnan(lyap_val):
        print(f"  Показатель Ляпунова (λ):  {lyap_val:.6f}")
    if not np.isnan(kse_val):
        print(f"  Энтропия Колмогорова-Синая: {kse_val:.6f} бит/шаг")

if __name__ == "__main__":
    print("ЗАПУСК ЭКСПЕРИМЕНТА: ОКНО СКОЛЬЗЯЩЕЙ СРЕДНЕЙ = 5 ДНЕЙ")
    
    asset, traders = run_simulation_window_5(days=1000)
    
    # Построение графиков
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # График цены
    axes[0, 0].plot(asset.price_history, 'b-', linewidth=1)
    axes[0, 0].set_title('Цена актива (window=5)', fontsize=12)
    axes[0, 0].set_xlabel('День')
    axes[0, 0].set_ylabel('Цена')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Графики богатства
    axes[0, 1].plot(wealth_history['days'], wealth_history['noise_avg'], 'b-', linewidth=2)
    axes[0, 1].axhline(y=200000, color='r', linestyle='--', alpha=0.5)
    axes[0, 1].set_title('Среднее богатство шумовых трейдеров', fontsize=12)
    axes[0, 1].set_xlabel('День')
    axes[0, 1].set_ylabel('Среднее богатство')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].plot(wealth_history['days'], wealth_history['chartist_avg'], 'g-', linewidth=2)
    axes[1, 0].axhline(y=250000, color='r', linestyle='--', alpha=0.5)
    axes[1, 0].set_title(f'Среднее богатство технических трейдеров (window=5)', fontsize=12)
    axes[1, 0].set_xlabel('День')
    axes[1, 0].set_ylabel('Среднее богатство')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(wealth_history['days'], wealth_history['fundamentalist_avg'], 'r-', linewidth=2)
    axes[1, 1].axhline(y=1050000, color='r', linestyle='--', alpha=0.5)
    axes[1, 1].set_title('Среднее богатство фундаменталистов', fontsize=12)
    axes[1, 1].set_xlabel('День')
    axes[1, 1].set_ylabel('Среднее богатство')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../results/window_5_analysis.png', dpi=150)
    plt.show()
    
    # Гистограммы распределения богатства
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    noise_wealth = [t.wealth_history[-1] for t in traders if t.type == 'noise']
    chartist_wealth = [t.wealth_history[-1] for t in traders if t.type == 'chartist']
    fundamentalist_wealth = [t.wealth_history[-1] for t in traders if t.type == 'fundamentalist']
    
    axes[0].hist(noise_wealth, bins=20, color='blue', alpha=0.7, edgecolor='black')
    axes[0].set_title('Распределение богатства\nшумовых трейдеров', fontsize=12)
    axes[0].set_xlabel('Богатство')
    axes[0].set_ylabel('Количество трейдеров')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].hist(chartist_wealth, bins=20, color='green', alpha=0.7, edgecolor='black')
    axes[1].set_title(f'Распределение богатства\nтехнических трейдеров (window=5)', fontsize=12)
    axes[1].set_xlabel('Богатство')
    axes[1].set_ylabel('Количество трейдеров')
    axes[1].grid(True, alpha=0.3)
    
    axes[2].hist(fundamentalist_wealth, bins=20, color='red', alpha=0.7, edgecolor='black')
    axes[2].set_title('Распределение богатства\nфундаменталистов', fontsize=12)
    axes[2].set_xlabel('Богатство')
    axes[2].set_ylabel('Количество трейдеров')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
