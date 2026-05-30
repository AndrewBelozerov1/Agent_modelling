import numpy as np
import matplotlib.pyplot as plt
from .asset import Asset
from .order_book import OrderBook
from .traders import NoiseTrader, ChartistTrader, FundamentalistTrader
from .metrics import calculate_hurst_exponent, calculate_lyapunov_exponent, calculate_kolmogorov_entropy

# Глобальный словарь для хранения истории богатства
wealth_history = {
    'days': [],
    'noise_avg': [],
    'chartist_avg': [],
    'fundamentalist_avg': []
}

def run_simulation(days=1000, noise_traders=50, chartists=20, fundamentalists=30, 
                   with_circuit_breaker=False, with_evolution=False):
    asset = Asset() if not with_circuit_breaker else AssetWithCircuitBreaker()
    order_book = OrderBook()
    traders = []
    
    for _ in range(noise_traders):
        traders.append(NoiseTrader(cash=100000, assets=1000))
    for _ in range(chartists):
        traders.append(ChartistTrader(cash=150000, assets=1000, window=20, alpha=0.1))
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
        
        # Эволюция трейдеров (каждые 100 дней)
        if with_evolution and day % 100 == 0 and day > 0:
            traders = _evolve_traders(traders, day)
        
        trading_halted = asset.check_halt_status() if with_circuit_breaker else False
        
        if not trading_halted:
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
            _log_simulation_status(day, asset, traders, noise_traders, chartists, fundamentalists, trading_halted)
    
    # Финальный анализ
    _print_final_analysis(traders, initial_wealths, asset.price_history, 
                          noise_traders, chartists, fundamentalists)
    
    return asset, traders

def _evolve_traders(traders, day):
    print(f"\nЭволюция трейдеров на день {day}")
    noise_traders_list = [t for t in traders if t.type == 'noise' and t.active]
    if len(noise_traders_list) >= 2:
        traders_to_evolve = np.random.choice(noise_traders_list, 2, replace=False)
        for trader in traders_to_evolve:
            from .traders import ChartistTrader
            new_chartist = ChartistTrader(cash=trader.cash, assets=trader.assets, window=20, alpha=0.1)
            new_chartist.wealth_history = trader.wealth_history.copy()
            traders.append(new_chartist)
            trader.active = False
            print(f"Шумовой трейдер эволюционировал в чартиста")
    
    chartist_traders_list = [t for t in traders if t.type == 'chartist' and t.active]
    if len(chartist_traders_list) >= 1:
        from .traders import FundamentalistTrader
        trader_to_evolve = np.random.choice(chartist_traders_list, 1)[0]
        new_fundamentalist = FundamentalistTrader(cash=trader_to_evolve.cash, assets=trader_to_evolve.assets, gamma=2, rf=0.0003)
        new_fundamentalist.wealth_history = trader_to_evolve.wealth_history.copy()
        traders.append(new_fundamentalist)
        trader_to_evolve.active = False
        print(f"Чартист эволюционировал в фундаменталиста")
    
    return traders

def _log_simulation_status(day, asset, traders, noise_traders, chartists, fundamentalists, trading_halted):
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
    if trading_halted:
        print(f"Торги остановлены! Осталось дней: {asset.halt_remaining}")
    print(f"Active Traders - Noise: {noise_stats['active']}/{noise_traders}, "
          f"Chartists: {chart_stats['active']}/{chartists}, "
          f"Fundamentalists: {fund_stats['active']}/{fundamentalists}")
    
    if noise_stats['active'] > 0:
        print(f"Noise Traders: Avg Cash = {np.mean(noise_stats['cash']):.2f}, "
              f"Avg Assets = {np.mean(noise_stats['assets']):.2f}, "
              f"Avg Wealth = {wealth_history['noise_avg'][-1]:.2f}")
    if chart_stats['active'] > 0:
        print(f"Chartist Traders: Avg Cash = {np.mean(chart_stats['cash']):.2f}, "
              f"Avg Assets = {np.mean(chart_stats['assets']):.2f}, "
              f"Avg Wealth = {wealth_history['chartist_avg'][-1]:.2f}")
    if fund_stats['active'] > 0:
        print(f"Fundamentalist Traders: Avg Cash = {np.mean(fund_stats['cash']):.2f}, "
              f"Avg Assets = {np.mean(fund_stats['assets']):.2f}, "
              f"Avg Wealth = {wealth_history['fundamentalist_avg'][-1]:.2f}")

def _print_final_analysis(traders, initial_wealths, price_history, noise_traders, chartists, fundamentalists):
    print("\n" + "="*50)
    print("АНАЛИЗ РЕЗУЛЬТАТОВ СИМУЛЯЦИИ")
    print("="*50)
    
    profitable_noise = profitable_chartists = profitable_fundamentalists = 0
    
    for trader in traders:
        if id(trader) not in initial_wealths:
            continue
        if trader.wealth_history[-1] > initial_wealths[id(trader)]:
            if trader.type == 'noise':
                profitable_noise += 1
            elif trader.type == 'chartist':
                profitable_chartists += 1
            elif trader.type == 'fundamentalist':
                profitable_fundamentalists += 1
    
    print(f"Шумовые трейдеры, вышедшие в плюс: {profitable_noise}/{noise_traders} ({profitable_noise/noise_traders*100:.1f}%)")
    print(f"Технические трейдеры, вышедшие в плюс: {profitable_chartists}/{chartists} ({profitable_chartists/chartists*100:.1f}%)")
    print(f"Фундаменталисты, вышедшие в плюс: {profitable_fundamentalists}/{fundamentalists} ({profitable_fundamentalists/fundamentalists*100:.1f}%)")
    
    # Расчёт метрик
    hurst_val = calculate_hurst_exponent(np.array(price_history))
    lyap_val = calculate_lyapunov_exponent(np.array(price_history), emb_dim=10, lag=1)
    kse_val = calculate_kolmogorov_entropy(np.array(price_history), emb_dim=3)
    
    print(f"\nПОКАЗАТЕЛЬ ХЕРСТА (H) = {hurst_val:.4f}")
    print(f"СТАРШИЙ ПОКАЗАТЕЛЬ ЛЯПУНОВА (λ_max) = {lyap_val:.4f}")
    print(f"ЭНТРОПИЯ КОЛМОГОРОВА-СИНАЯ (h_KS) = {kse_val:.4f} бит/шаг")
