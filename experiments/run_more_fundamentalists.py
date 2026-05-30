import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.simulation import run_simulation
import matplotlib.pyplot as plt

if __name__ == "__main__":
    print("СЦЕНАРИЙ: БОЛЬШЕ ФУНДАМЕНТАЛИСТОВ (50 / 20 / 300)")
    
    asset, traders, wealth_history = run_simulation(
        days=1000,
        noise_traders=50,
        chartists=20,
        fundamentalists=300
    )
    
    plt.figure(figsize=(14, 7))
    plt.plot(asset.price_history, label='Price', color='blue')
    plt.xlabel('Day')
    plt.ylabel('Price')
    plt.title('Динамика цены актива (300 фундаменталистов)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()
