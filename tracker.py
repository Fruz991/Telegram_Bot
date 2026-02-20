import os
import json
from datetime import datetime, timezone
import logging

logger = logging.getLogger(__name__)

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
                with open(self.trades_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return {
                "daily_stops": 0,
                "daily_wins": 0,
                "last_reset": str(datetime.now(timezone.utc).date()),
                "history": []
            }
        except Exception as e:
            logger.error(f"Ошибка загрузки trades: {e}")
            return {
                "daily_stops": 0,
                "daily_wins": 0,
                "last_reset": str(datetime.now(timezone.utc).date()),
                "history": []
            }
    
    def save_trades(self):
        try:
            with open(self.trades_file, 'w', encoding='utf-8') as f:
                json.dump(self.trades, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Ошибка сохранения trades: {e}")
    
    def reset_daily_if_needed(self):
        # Используем UTC время для независимости от сервера
        today = str(datetime.now(timezone.utc).date())
        if self.trades["last_reset"] != today:
            self.trades["daily_stops"] = 0
            self.trades["daily_wins"] = 0
            self.trades["last_reset"] = today
            self.save_trades()
            logger.info("Счётчик сделок сброшен на новый день (UTC)")
    
    def add_stop(self):
        self.reset_daily_if_needed()
        self.trades["daily_stops"] += 1
        self.trades["history"].append({
            "date": datetime.now(timezone.utc).isoformat(),
            "result": "LOSS"
        })
        self.save_trades()
        logger.info("Добавлен STOP. Всего сегодня: %d", self.trades["daily_stops"])
    
    def add_win(self):
        self.reset_daily_if_needed()
        self.trades["daily_wins"] += 1
        self.trades["history"].append({
            "date": datetime.now(timezone.utc).isoformat(),
            "result": "WIN"
        })
        self.save_trades()
        logger.info("Добавлен WIN. Всего сегодня: %d", self.trades["daily_wins"])
    
    def get_stops_count(self):
        self.reset_daily_if_needed()
        return self.trades["daily_stops"]
    
    def get_wins_count(self):
        self.reset_daily_if_needed()
        return self.trades["daily_wins"]
    
    def can_trade(self):
        return self.get_stops_count() < 3
    
    def get_stats(self):
        self.reset_daily_if_needed()
        total = len(self.trades["history"])
        wins = self.trades["daily_wins"]
        losses = self.trades["daily_stops"]
        win_rate = (wins / (wins + losses) * 100) if (wins + losses) > 0 else 0
        
        return {
            "daily_stops": losses,
            "daily_wins": wins,
            "total_trades": total,
            "win_rate": round(win_rate, 2),
            "can_trade": self.can_trade()
        }

tracker = TradeTracker()