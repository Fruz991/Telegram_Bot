import os
import requests
import pandas as pd
import logging
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# =====================================================
# MACRO INDICATORS TRACKER (DXY + S&P 500)
# =====================================================
class MacroTracker:
    def __init__(self):
        self.api_key = os.getenv("TWELVE_DATA_API_KEY")
        if not self.api_key:
            logger.warning("TWELVE_DATA_API_KEY не найден! Макро данные отключены")
        
        self.base_url = "https://api.twelvedata.com"
        self.cache = {
            "spx": {"value": None, "trend": "FLAT", "timestamp": 0, "error_count": 0},
            "dxy": {"value": None, "trend": "FLAT", "timestamp": 0, "error_count": 0},
        }
        self.cache_duration = 1800  # 30 минут (для экономии лимита 800 запросов/день)
        self.max_retries = 3  # Максимум ошибок перед паузой
    
    def get_time_series(self, symbol, interval="1h", outputsize=50):
        """Получает исторические данные"""
        try:
            url = f"{self.base_url}/time_series"
            params = {
                "symbol": symbol,
                "interval": interval,
                "outputsize": outputsize,
                "apikey": self.api_key,
                "format": "JSON"
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if "values" not in data:
                logger.error(f"Нет данных для {symbol}: {data}")
                return None
            
            # Преобразуем в DataFrame
            df = pd.DataFrame(data["values"])
            df["datetime"] = pd.to_datetime(df["datetime"])
            df = df.astype({
                "open": float,
                "high": float,
                "low": float,
                "close": float,
                "volume": float
            })
            df = df.sort_values("datetime")
            
            return df
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                logger.warning("Превышен лимит запросов Twelve Data")
            else:
                logger.error(f"HTTP ошибка {symbol}: {e}")
            return None
        except Exception as e:
            logger.error(f"Ошибка получения {symbol}: {e}")
            return None
    
    def analyze_trend(self, df):
        """Анализирует тренд по EMA"""
        if df is None or len(df) < 50:
            return "FLAT"
        
        df = df.copy()
        df['EMA20'] = df['close'].ewm(span=20).mean()
        df['EMA50'] = df['close'].ewm(span=50).mean()
        
        last = df.iloc[-1]
        
        if pd.isna(last['EMA20']) or pd.isna(last['EMA50']):
            return "FLAT"
        
        if last['close'] > last['EMA20'] > last['EMA50']:
            return "STRONG"  # Восходящий тренд
        elif last['close'] < last['EMA20'] < last['EMA50']:
            return "WEAK"   # Нисходящий тренд
        else:
            return "FLAT"
    
    def get_spx_data(self):
        """Получает данные S&P 500 через ETF (SPY)"""
        # Бесплатный тариф: используем SPY вместо SPX
        return self.get_time_series("SPY")
    
    def get_dxy_data(self):
        """Получает данные DXY через ETF (UUP)"""
        # Бесплатный тариф: используем UUP вместо DXY
        return self.get_time_series("UUP")
    
    def get_spx_cached(self):
        """Кэш для SPX"""
        now = datetime.now().timestamp()
        cache_age = now - self.cache["spx"]["timestamp"]
        
        # Если много ошибок — увеличиваем паузу
        if self.cache["spx"]["error_count"] >= self.max_retries:
            if cache_age < 1800:  # 30 минут пауза после ошибок
                logger.debug(f"SPX: пауза после ошибок (остаток {1800 - cache_age:.0f} сек)")
                return self.cache["spx"]
            else:
                self.cache["spx"]["error_count"] = 0  # Сброс после паузы
        
        if cache_age < self.cache_duration:
            logger.debug(f"SPX: возвращаем кэш (возраст {cache_age:.0f} сек)")
            return self.cache["spx"]

        logger.info("SPX: обновляем данные...")
        df = self.get_spx_data()
        if df is None or len(df) == 0:
            self.cache["spx"]["error_count"] += 1
            logger.warning(f"SPX: не удалось получить данные (ошибок: {self.cache['spx']['error_count']})")
            return {"value": None, "trend": "ERROR", "change": 0, "error_count": self.cache["spx"]["error_count"]}

        trend = self.analyze_trend(df)
        last_value = df.iloc[-1]['close']
        prev_value = df.iloc[-2]['close'] if len(df) > 1 else last_value
        change = ((last_value - prev_value) / prev_value) * 100

        self.cache["spx"] = {
            "value": last_value,
            "trend": trend,
            "change": change,
            "df": df,
            "timestamp": now,
            "error_count": 0
        }

        logger.info(f"SPY (S&P 500) обновлён: {last_value:.2f} ({trend}, {change:+.2f}%)")
        return self.cache["spx"]
    
    def get_dxy_cached(self):
        """Кэш для DXY"""
        now = datetime.now().timestamp()
        cache_age = now - self.cache["dxy"]["timestamp"]
        
        # Если много ошибок — увеличиваем паузу
        if self.cache["dxy"]["error_count"] >= self.max_retries:
            if cache_age < 1800:  # 30 минут пауза после ошибок
                logger.debug(f"DXY: пауза после ошибок (остаток {1800 - cache_age:.0f} сек)")
                return self.cache["dxy"]
            else:
                self.cache["dxy"]["error_count"] = 0  # Сброс после паузы
        
        if cache_age < self.cache_duration:
            logger.debug(f"DXY: возвращаем кэш (возраст {cache_age:.0f} сек)")
            return self.cache["dxy"]

        logger.info("DXY: обновляем данные...")
        df = self.get_dxy_data()
        if df is None or len(df) == 0:
            self.cache["dxy"]["error_count"] += 1
            logger.warning(f"DXY: не удалось получить данные (ошибок: {self.cache['dxy']['error_count']})")
            return {"value": None, "trend": "ERROR", "change": 0, "error_count": self.cache["dxy"]["error_count"]}

        trend = self.analyze_trend(df)
        last_value = df.iloc[-1]['close']
        prev_value = df.iloc[-2]['close'] if len(df) > 1 else last_value
        change = ((last_value - prev_value) / prev_value) * 100

        self.cache["dxy"] = {
            "value": last_value,
            "trend": trend,
            "change": change,
            "df": df,
            "timestamp": now,
            "error_count": 0
        }

        logger.info(f"UUP (DXY) обновлён: {last_value:.2f} ({trend}, {change:+.2f}%)")
        return self.cache["dxy"]
    
    def get_crypto_impact(self, spx, dxy):
        """Определяет влияние макро на крипту"""
        spx_trend = spx.get("trend", "FLAT")
        dxy_trend = dxy.get("trend", "FLAT")
        
        if spx_trend == "STRONG" and dxy_trend == "WEAK":
            return "BULLISH"  # 🟢 Лучшая ситуация
        elif spx_trend == "WEAK" and dxy_trend == "STRONG":
            return "BEARISH"  # 🔴 Худшая ситуация
        else:
            return "NEUTRAL"  # 😐 Нейтрально
    
    def get_market_context(self):
        """Возвращает полный макро контекст"""
        spx = self.get_spx_cached()
        dxy = self.get_dxy_cached()
        crypto_impact = self.get_crypto_impact(spx, dxy)
        
        logger.info(f"Макро: SPX {spx.get('value', 'N/A'):.2f} ({spx.get('trend', 'N/A')}), DXY {dxy.get('value', 'N/A'):.2f} ({dxy.get('trend', 'N/A')})")

        return {
            "spx": spx,
            "dxy": dxy,
            "crypto_impact": crypto_impact
        }

# Глобальный экземпляр
macro_tracker = MacroTracker()
