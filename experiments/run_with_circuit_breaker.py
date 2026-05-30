import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Импорт классов из src
from src.asset import Asset
from src.order_book import OrderBook
from src.traders import NoiseTrader, ChartistTrader, FundamentalistTrader
from src.metrics import calculate_hurst_exponent, calculate_lyapunov_exponent, calculate_kolmogorov_entropy


# Глобальная переменная для хранения истории богатства
wealth_history = {
    'days': [],
    'noise_avg': [],
    'chartist_avg': [],
    'fundamentalist_avg': []
}


def run_simulation_with_circuit_breaker(days=1000, noise_traders=50, chartists=20, fundamentalists=30,
                                         halt_threshold=0.20, halt_duration=5):
    
    asset = Asset(initial_price=100, halt_threshold=halt_threshold, halt_duration=halt_duration)
    order_book = OrderBook()
    traders = []
    
    for _ in range(noise_traders):
        traders.append(NoiseTrader(cash=100000, assets=1000))
    for _ in range(chartists):
        traders.append(ChartistTrader(cash=150000, assets=1000, window=20, alpha=0.1))
    for _ in range(fundamentalists):
        traders.append(FundamentalistTrader(cash=1000000, assets=500, gamma=2, rf=0.0003))

    for _ in range(20):
        new_price = asset.current_price * np.exp(0.0003 - 0.02**2/2 + 0.02 * np.random.randn())
        asset.update_price(new_price)
        asset.update_dividend()
    
    initial_wealths = {}
    for trader in traders:
        initial_wealths[id(trader)] = trader.wealth_history[0]

    wealth_history['days'] = []
    wealth_history['noise_avg'] = []
    wealth_history['chartist_avg'] = []
    wealth_history['fundamentalist_avg'] = []
    
    halt_events = []  # Список дней, когда происходила остановка

    for day in range(days):
        asset.update_dividend()
        trading_halted = asset.check_halt_status()
        if not trading_halted:
            for trader in traders:
                if trader.cash <= 0 and trader.assets <= 0:
                    trader.active = False
                    continue
                order_type, price, qty = trader.make_decision(asset)
                if order_type and price and qty:
                    order_book.add_order(order_type, price, qty, id(trader))
            executed = order_book.match_orders(traders, asset)
        else:
            # Если торги остановлены, записываем событие
            if day not in halt_events:
                halt_events.append(day)
        for trader in traders:
            trader.update_wealth(asset.current_price)

        if day % 100 == 0 or day == days - 1:
            noise_wealth = []
            chartist_wealth = []
            fundamentalist_wealth = []
            
            for trader in traders:
                if trader.type == 'noise' and trader.active:
                    current_wealth = trader.cash + trader.assets * asset.current_price
                    noise_wealth.append(current_wealth)
                elif trader.type == 'chartist' and trader.active:
                    current_wealth = trader.cash + trader.assets * asset.current_price
                    chartist_wealth.append(current_wealth)
                elif trader.type == 'fundamentalist' and trader.active:
                    current_wealth = trader.cash + trader.assets * asset.current_price
                    fundamentalist_wealth.append(current_wealth)
            
            wealth_history['days'].append(day)
            wealth_history['noise_avg'].append(np.mean(noise_wealth) if noise_wealth else 0)
            wealth_history['chartist_avg'].append(np.mean(chartist_wealth) if chartist_wealth else 0)
            wealth_history['fundamentalist_avg'].append(np.mean(fundamentalist_wealth) if fundamentalist_wealth else 0)
            
            noise_stats = {'cash': [], 'assets': [], 'active': 0}
            chart_stats = {'cash': [], 'assets': [], 'active': 0}
            fund_stats = {'cash': [], 'assets': [], 'active': 0}
            
            for trader in traders:
                if trader.type == 'noise' and trader.active:
                    noise_stats['cash'].append(trader.cash)
                    noise_stats['assets'].append(trader.assets)
                    noise_stats['active'] += 1
                elif trader.type == 'chartist' and trader.active:
                    chart_stats['cash'].append(trader.cash)
                    chart_stats['assets'].append(trader.assets)
                    chart_stats['active'] += 1
                elif trader.type == 'fundamentalist' and trader.active:
                    fund_stats['cash'].append(trader.cash)
                    fund_stats['assets'].append(trader.assets)
                    fund_stats['active'] += 1
            
            print(f"\n{'='*60}")
            print(f"День {day}: Цена = {asset.current_price:.2f}")
            if trading_halted:
                print(f"ТОРГИ ОСТАНОВЛЕНЫ! Осталось дней: {asset.halt_remaining}")
            print(f"Активные трейдеры - Шумовые: {noise_stats['active']}/{noise_traders}, "
                  f"Технические: {chart_stats['active']}/{chartists}, "
                  f"Фундаменталисты: {fund_stats['active']}/{fundamentalists}")
            
            if noise_stats['active'] > 0:
                print(f"Шумовые трейдеры: Средние наличные = {np.mean(noise_stats['cash']):.2f}, "
                      f"Средние акции = {np.mean(noise_stats['assets']):.2f}, "
                      f"Среднее богатство = {wealth_history['noise_avg'][-1]:.2f}")
            if chart_stats['active'] > 0:
                print(f"Технические трейдеры: Средние наличные = {np.mean(chart_stats['cash']):.2f}, "
                      f"Средние акции = {np.mean(chart_stats['assets']):.2f}, "
                      f"Среднее богатство = {wealth_history['chartist_avg'][-1]:.2f}")
            if fund_stats['active'] > 0:
                print(f"Фундаменталисты: Средние наличные = {np.mean(fund_stats['cash']):.2f}, "
                      f"Средние акции = {np.mean(fund_stats['assets']):.2f}, "
                      f"Среднее богатство = {wealth_history['fundamentalist_avg'][-1]:.2f}")
    
    print("АНАЛИЗ РЕЗУЛЬТАТОВ СИМУЛЯЦИИ (CIRCUIT BREAKER)")
    
    if halt_events:
        print(f"\n Произошло {len(halt_events)} остановок торгов в дни: {halt_events}")
    
    profitable_noise = 0
    profitable_chartists = 0
    profitable_fundamentalists = 0
    total_profitable = 0
    
    for trader in traders:
        initial_wealth = initial_wealths[id(trader)]
        final_wealth = trader.wealth_history[-1]
        is_profitable = final_wealth > initial_wealth
        if is_profitable:
            total_profitable += 1
            if trader.type == 'noise':
                profitable_noise += 1
            elif trader.type == 'chartist':
                profitable_chartists += 1
            elif trader.type == 'fundamentalist':
                profitable_fundamentalists += 1
    
    total_traders = len(traders)
    print(f"\n Прибыльность трейдеров:")
    print(f"  Шумовые трейдеры в плюсе: {profitable_noise}/{noise_traders} ({profitable_noise/noise_traders*100:.1f}%)")
    print(f"  Технические трейдеры в плюсе: {profitable_chartists}/{chartists} ({profitable_chartists/chartists*100:.1f}%)")
    print(f"  Фундаменталисты в плюсе: {profitable_fundamentalists}/{fundamentalists} ({profitable_fundamentalists/fundamentalists*100:.1f}%)")
    print(f"  Общий процент трейдеров в плюсе: {total_profitable}/{total_traders} ({total_profitable/total_traders*100:.1f}%)")

    noise_wealth_change = []
    chartist_wealth_change = []
    fundamentalist_wealth_change = []
    
    for trader in traders:
        initial_wealth = initial_wealths[id(trader)]
        final_wealth = trader.wealth_history[-1]
        wealth_change = ((final_wealth - initial_wealth) / initial_wealth) * 100
        if trader.type == 'noise':
            noise_wealth_change.append(wealth_change)
        elif trader.type == 'chartist':
            chartist_wealth_change.append(wealth_change)
        elif trader.type == 'fundamentalist':
            fundamentalist_wealth_change.append(wealth_change)
    
    print(f"\n Среднее изменение богатства:")
    if noise_wealth_change:
        print(f"  Шумовые трейдеры: {np.mean(noise_wealth_change):.2f}%")
    if chartist_wealth_change:
        print(f"  Технические трейдеры: {np.mean(chartist_wealth_change):.2f}%")
    if fundamentalist_wealth_change:
        print(f"  Фундаменталисты: {np.mean(fundamentalist_wealth_change):.2f}%")
    
    print(f"\n НЕЛИНЕЙНЫЕ МЕТРИКИ:")
    price_series = np.array(asset.price_history)
    
    hurst_val = calculate_hurst_exponent(price_series)
    if not np.isnan(hurst_val):
        print(f"  Показатель Херста (H) = {hurst_val:.4f}")
    else:
        print("  Показатель Херста: не удалось рассчитать")
    
    lyap_val = calculate_lyapunov_exponent(price_series, emb_dim=10, lag=1)
    if not np.isnan(lyap_val):
        print(f"  Старший показатель Ляпунова (λ_max) = {lyap_val:.4f}")
        print(f"  Горизонт планирования = {1/lyap_val:.1f} дней")
    else:
        print("  Показатель Ляпунова: не удалось рассчитать")
    
    kse_val = calculate_kolmogorov_entropy(price_series, emb_dim=3)
    if not np.isnan(kse_val):
        print(f"  Энтропия Колмогорова-Синая (h_KS) = {kse_val:.4f} бит/шаг")
    else:
        print("  Энтропия Колмогорова-Синая: не удалось рассчитать")
    
    # Сохраняем историю остановок в asset для последующей визуализации
    asset.halt_events = halt_events
    
    return asset, traders, wealth_history


if __name__ == "__main__":
    print("ЗАПУСК СИМУЛЯЦИИ С CIRCUIT BREAKER")
    print(f"Порог остановки: изменение цены > 20% за день")
    print(f"Длительность остановки: 5 дней")
    
    asset, traders, wealth_history = run_simulation_with_circuit_breaker(
        days=1000,
        noise_traders=50,
        chartists=20,
        fundamentalists=30,
        halt_threshold=0.20,
        halt_duration=5
    )

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    axes[0].plot(asset.price_history, label='Цена актива', color='purple', linewidth=1)
    for halt_day in asset.halt_events:
        axes[0].axvline(x=halt_day, color='red', linestyle='--', alpha=0.7, linewidth=1)
    axes[0].set_xlabel('День')
    axes[0].set_ylabel('Цена')
    axes[0].set_title('Динамика цены актива с Circuit Breaker (красные линии = остановки торгов)')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    axes[1].plot(wealth_history['days'], wealth_history['noise_avg'], 'b-', linewidth=2, label='Шумовые трейдеры')
    axes[1].plot(wealth_history['days'], wealth_history['chartist_avg'], 'g-', linewidth=2, label='Технические трейдеры')
    axes[1].plot(wealth_history['days'], wealth_history['fundamentalist_avg'], 'r-', linewidth=2, label='Фундаменталисты')
    axes[1].set_xlabel('День')
    axes[1].set_ylabel('Среднее богатство')
    axes[1].set_title('Динамика среднего богатства трейдеров')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    plt.tight_layout()
    plt.show()
