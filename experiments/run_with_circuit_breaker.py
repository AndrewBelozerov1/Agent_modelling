import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
from src.simulation import run_simulation, wealth_history

if __name__ == "__main__":
    asset, traders = run_simulation(days=1000, noise_traders=50, chartists=20, 
                                     fundamentalists=30, with_circuit_breaker=True)
    
    plt.figure(figsize=(14, 7))
    plt.plot(asset.price_history, label='Price with Circuit Breaker')
    plt.xlabel('Day')
    plt.ylabel('Price')
    plt.grid()
    plt.legend()
    plt.title('Сценарий с Circuit Breaker (остановка торгов при движении >20%)')
    plt.show()
