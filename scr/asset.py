import numpy as np

class Asset:
    def __init__(self, initial_price=100):
        self.price_history = [initial_price]
        self.dividend_history = [0.016]
        self.current_price = initial_price
        self.trading_halted = False
        self.halt_threshold = 0.20
        self.halt_duration = 5
        self.halt_remaining = 0
        self.last_check_price = initial_price

    def update_price(self, new_price):
        if not self.trading_halted:
            price_change = abs(new_price / self.last_check_price - 1)
            if price_change > self.halt_threshold:
                self.trading_halted = True
                self.halt_remaining = self.halt_duration
                print(f"\nCircuit Breaker: Торги остановлены на {self.halt_duration} дней (изменение {price_change:.2%} > {self.halt_threshold:.2%})")
                return False

        self.current_price = new_price
        self.price_history.append(new_price)
        return True

    def update_dividend(self, rd=0.0008, sigma_d=0.0008):
        if np.random.rand() < 0.01:
            shock = np.random.randn() * 0.005
        else:
            shock = 0
        new_div = self.dividend_history[-1] * (1 + rd + sigma_d * np.random.randn() + shock)
        self.dividend_history.append(max(new_div, 0))
        return self.dividend_history[-1]

    def check_halt_status(self):
        if self.trading_halted:
            self.halt_remaining -= 1
            if self.halt_remaining <= 0:
                self.trading_halted = False
                self.last_check_price = self.current_price
                print("\nCircuit Breaker: Торги возобновлены")
        else:
            if len(self.price_history) > 0:
                self.last_check_price = self.price_history[-1]

        return self.trading_halted
