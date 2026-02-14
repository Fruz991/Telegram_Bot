import os
import json
from datetime import datetime


# =====================================================
# СИСТЕМА УЧЕТА СДЕЛОК И ЛИМИТОВ
# =====================================================
class TradeTracker:
    def __init__(self):
        self.trades_file = "trades_history.json"
        self.trades = self.load_trades()

    def load_trades(self):
        try:
            if os.path.exists(self.trades_file):
                with open(self.trades_file, 'r') as f:
                    return json.load(f)
            return {"daily_stops": 0, "last_reset": str(datetime.now().date()), "history": []}
        except:
            return {"daily_stops": 0, "last_reset": str(datetime.now().date()), "history": []}

    def save_trades(self):
        try:
            with open(self.trades_file, 'w') as f:
                json.dump(self.trades, f, indent=2)
        except Exception as e:
            print(f"Ошибка сохранения trades: {e}")

    def reset_daily_if_needed(self):
        today = str(datetime.now().date())
        if self.trades["last_reset"] != today:
            self.trades["daily_stops"] = 0
            self.trades["last_reset"] = today
            self.save_trades()

    def add_stop(self):
        self.reset_daily_if_needed()
        self.trades["daily_stops"] += 1
        self.save_trades()

    def get_stops_count(self):
        self.reset_daily_if_needed()
        return self.trades["daily_stops"]

    def can_trade(self):
        return self.get_stops_count() < 3


tracker = TradeTracker()