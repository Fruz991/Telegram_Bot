# =====================================================
# НАСТРОЙКИ (не секретные данные)
# =====================================================
import os
from dotenv import load_dotenv

# ВАЖНО: загружаем .env перед чтением переменных!
load_dotenv()

# =====================================================
# API КЛЮЧИ
# =====================================================
TWELVE_DATA_API_KEY = os.getenv("TWELVE_DATA_API_KEY")

# =====================================================
# СПИСОК МОНЕТ И ТАЙМФРЕЙМЫ
# =====================================================
SYMBOLS = [
    "BTC/USDT", "SOL/USDT", "ETH/USDT", "SUI/USDT", "LTC/USDT",
    "BNB/USDT", "WIF/USDT", "ADA/USDT", "ATOM/USDT",
    "ENA/USDT", "NEAR/USDT", "OP/USDT",
    "AVAX/USDT", "LINK/USDT", "ZRO/USDT", "XRP/USDT", "FARTCOIN/USDT"
]
TIMEFRAMES = ['15m', '30m', '1h', '4h', '1d']
COOLDOWN_SECONDS = 1800  # 30 минут между сигналами на одну монету

# Настройки скана
SCAN_INTERVAL_SECONDS = 300  # 5 минут между полными сканами

# =====================================================
# НОВОСТИ FMP (Макроэкономика) - ЧИТАЕМ ИЗ ENV!
# =====================================================
FMP_API_KEY = os.getenv("FMP_API_KEY")  # ← Теперь читает из окружения!
FMP_CHECK_MINUTES = int(os.getenv("FMP_CHECK_MINUTES", 60))
FMP_IMPACT_FILTER = os.getenv("FMP_IMPACT_FILTER", "High,Medium").split(",")

# =====================================================
# НОВОСТИ RSS (Крипто)
# =====================================================
RSS_CHECK_MINUTES = int(os.getenv("RSS_CHECK_MINUTES", 30))

# Читаем из env, если есть — иначе дефолтные значения
rss_feeds_raw = os.getenv("RSS_FEEDS")
RSS_FEEDS = rss_feeds_raw.split(",") if rss_feeds_raw else [
    "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "https://cointelegraph.com/rss",
    "https://cryptoslate.com/feed/"
]

rss_keywords_raw = os.getenv("RSS_KEYWORDS")
RSS_KEYWORDS = rss_keywords_raw.split(",") if rss_keywords_raw else [
    "hack", "exploit", "SEC", "ban", "ETF", "listing", "delist"
]