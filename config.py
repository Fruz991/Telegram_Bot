# =====================================================
# НАСТРОЙКИ (не секретные данные)
# =====================================================
SYMBOLS = [
    "BTC/USDT", "SOL/USDT", "ETH/USDT", "SUI/USDT", "LTC/USDT",
    "BNB/USDT", "WIF/USDT", "ADA/USDT", "ATOM/USDT", "ZEC/USDT",
    "ENA/USDT", "NEAR/USDT", "OP/USDT"
]
TIMEFRAMES = ['15m', '30m', '1h', '4h', '1d']
COOLDOWN_SECONDS = 1800  # 30 минут между сигналами на одну монету

# Настройки скана
SCAN_INTERVAL_SECONDS = 300  # 5 минут между полными сканами

# Новости FMP (Макроэкономика)
FMP_API_KEY = ""  # Вставь ключ в .env
FMP_CHECK_MINUTES = 60  # Не торговать за 60 минут до макро-событий
FMP_IMPACT_FILTER = ["High", "Medium"]  # Фильтр по важности

# Новости RSS (Крипто)
RSS_CHECK_MINUTES = 30  # Не торговать за 30 минут до крипто-новостей
RSS_FEEDS = [
    "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "https://cointelegraph.com/rss",
    "https://cryptoslate.com/feed/"
]
RSS_KEYWORDS = ["hack", "exploit", "SEC", "ban", "ETF", "listing", "delist"]