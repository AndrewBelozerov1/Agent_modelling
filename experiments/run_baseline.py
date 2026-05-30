import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
from src import run_simulation, wealth_history

if __name__ == "__main__":
    asset, traders = run_simulation(days=1000, noise_traders=50, chartists=20, fundamentalists=30)
    
    # График цены
    plt.figure(figsize=(14, 7))
    plt.plot(asset.price_history, label='Price')
    plt.xlabel('Day')
    plt.ylabel('Price')
    plt.grid()
    plt.legend()
    plt.title('Базовый сценарий: цена актива')
    plt.savefig('../results/baseline_price.png')
    plt.show()
    
    # Графики богатства
    plt.figure(figsize=(14, 8))
    
    plt.subplot(3, 1, 1)
    plt.plot(wealth_history['days'], wealth_history['noise_avg'], 'b-', linewidth=2)
    plt.title('Изменение среднего богатства шумовых трейдеров')
    plt.xlabel('День')
    plt.ylabel('Среднее богатство')
    plt.grid(True)
    plt.axhline(y=200000, color='r', linestyle='--', alpha=0.5, label='Начальное богатство')
    plt.legend()
    
    plt.subplot(3, 1, 2)
    plt.plot(wealth_history['days'], wealth_history['chartist_avg'], 'g-', linewidth=2)
    plt.title('Изменение среднего богатства технических трейдеров')
    plt.xlabel('День')
    plt.ylabel('Среднее богатство')
    plt.grid(True)
    plt.axhline(y=250000, color='r', linestyle='--', alpha=0.5, label='Начальное богатство')
    plt.legend()
    
    plt.subplot(3, 1, 3)
    plt.plot(wealth_history['days'], wealth_history['fundamentalist_avg'], 'r-', linewidth=2)
    plt.title('Изменение среднего богатства фундаменталистов')
    plt.xlabel('День')
    plt.ylabel('Среднее богатство')
    plt.grid(True)
    plt.axhline(y=1050000, color='r', linestyle='--', alpha=0.5, label='Начальное богатство')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('../results/baseline_wealth.png')
    plt.show()
