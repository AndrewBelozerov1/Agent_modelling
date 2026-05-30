import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.simulation import run_simulation
import matplotlib.pyplot as plt

if __name__ == "__main__":
    print("СЦЕНАРИЙ: БОЛЬШЕ ТЕХНИЧЕСКИХ ТРЕЙДЕРОВ (25 / 130 / 15)")
    
    asset, traders, wealth_history = run_simulation(
        days=1000,
        noise_traders=25,
        chartists=130,
        fundamentalists=15
    )
    
    plt.figure(figsize=(14, 7))
    plt.plot(asset.price_history, label='Price', color='green')
    plt.xlabel('Day')
    plt.ylabel('Price')
    plt.title('Динамика цены актива (130 технических трейдеров)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()
