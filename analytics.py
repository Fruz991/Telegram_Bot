import asyncio
import os
import socket
import logging
import aiohttp
import feedparser
import ccxt
import pandas as pd
import numpy as np
import ta
import pytz
from datetime import datetime, timezone, timedelta
from config import TIMEFRAMES, FMP_API_KEY, FMP_CHECK_MINUTES, FMP_IMPACT_FILTER
from config import RSS_CHECK_MINUTES, RSS_FEEDS, RSS_KEYWORDS

logger = logging.getLogger(__name__)

# =====================================================
# IPv4
# =====================================================
original_getaddrinfo = socket.getaddrinfo

def getaddrinfo_ipv4(*args, **kwargs):
    return [x for x in original_getaddrinfo(*args, **kwargs) if x[0].name == 'AF_INET']

socket.getaddrinfo = getaddrinfo_ipv4

# =====================================================
# БИРЖА
# =====================================================
exchange = ccxt.bybit({
    "apiKey": os.getenv("BYBIT_API_KEY"),
    "secret": os.getenv("BYBIT_API_SECRET"),
    "enableRateLimit": True,
    "rateLimit": 50,
    "timeout": 10000,
    "options": {
        "defaultType": "future",
        "adjustForTimeDifference": True
    }
})

# =====================================================
# ПРОВЕРКА МАКРО-СОБЫТИЙ (FMP) - ОТКЛЮЧЕНО
# =====================================================
async def check_fmp_calendar_blocking():
    """Проверяет экономический календарь FMP - ОТКЛЮЧЕНО"""
    # ОТКЛЮЧЕНО: API endpoint больше не доступен для бесплатных аккаунтов
    # Legacy endpoints were deprecated in August 2025
    logger.info("FMP проверка отключена (API endpoint устарел)")
    return False

# =====================================================
# ПРОВЕРКА RSS НОВОСТЕЙ
# =====================================================
async def check_rss_blocking():
    """Проверяет RSS-фиды на наличие важных новостей"""
    if not RSS_FEEDS:
        return False
    
    now_utc = datetime.now(timezone.utc)
    
    for feed_url in RSS_FEEDS:
        try:
            loop = asyncio.get_running_loop()
            feed = await loop.run_in_executor(None, feedparser.parse, feed_url)
            
            for entry in feed.entries[:10]:
                published = entry.get("published_parsed")
                if published:
                    pub_time = datetime(*published[:6], tzinfo=timezone.utc)
                    time_diff = now_utc - pub_time
                    
                    if time_diff.total_seconds() < RSS_CHECK_MINUTES * 60:
                        title = (entry.get("title", "") + " " + entry.get("summary", "")).lower()
                        if any(kw.lower() in title for kw in RSS_KEYWORDS):
                            logger.warning(f"RSS новость: {entry.get('title')}")
                            return True
        except Exception as e:
            logger.error(f"Ошибка RSS {feed_url}: {e}")
            continue
    
    return False

# =====================================================
# ОБЪЕДИНЁННАЯ ПРОВЕРКА НОВОСТЕЙ
# =====================================================
async def check_news_blocking():
    """Главная функция проверки новостей"""
    # FMP отключён, проверяем только RSS
    rss_blocking = await check_rss_blocking()
    if rss_blocking:
        return True
    
    return False

# =====================================================
# ТОРГОВЫЙ ПЛАН
# =====================================================
def build_advanced_trade_plan(price, atr, side):
    risk = atr * 1.5
    if side == "LONG":
        entry_min = price - (atr * 0.2)
        entry_max = price + (atr * 0.2)
        stop_loss = price - risk
        invalidation = stop_loss - (atr * 0.3)
        tp1 = price + (risk * 1.5)
        tp2 = price + (risk * 2.0)
        tp3 = price + (risk * 3.0)
    else:
        entry_min = price - (atr * 0.2)
        entry_max = price + (atr * 0.2)
        stop_loss = price + risk
        invalidation = stop_loss + (atr * 0.3)
        tp1 = price - (risk * 1.5)
        tp2 = price - (risk * 2.0)
        tp3 = price - (risk * 3.0)
    
    return {
        "entry_min": entry_min,
        "entry_max": entry_max,
        "stop_loss": stop_loss,
        "invalidation": invalidation,
        "tp1": tp1,
        "tp2": tp2,
        "tp3": tp3,
    }

# =====================================================
# УРОВНИ ПОДДЕРЖКИ/СОПРОТИВЛЕНИЯ
# =====================================================
def find_support_resistance(df, window=10, num_levels=3):
    highs = df['high'].values
    lows = df['low'].values
    levels = []
    
    for i in range(window, len(df) - window):
        if highs[i] == max(highs[i - window:i + window]):
            levels.append(("resistance", highs[i]))
        if lows[i] == min(lows[i - window:i + window]):
            levels.append(("support", lows[i]))
    
    clustered = []
    used = set()
    price_range = (df['high'].max() - df['low'].min()) * 0.01
    
    for i in range(len(levels)):
        if i in used:
            continue
        type1, price1 = levels[i]
        cluster = [price1]
        for j in range(len(levels)):
            if j != i and j not in used:
                type2, price2 = levels[j]
                if abs(price1 - price2) < price_range:
                    cluster.append(price2)
                    used.add(j)
        used.add(i)
        avg_price = sum(cluster) / len(cluster)
        clustered.append((type1, avg_price, len(cluster)))
    
    clustered.sort(key=lambda x: x[2], reverse=True)
    current_price = df['close'].iloc[-1]
    
    supports = sorted(
        [(t, p) for t, p, _ in clustered if p < current_price],
        key=lambda x: x[1], reverse=True
    )[:num_levels]
    
    resistances = sorted(
        [(t, p) for t, p, _ in clustered if p > current_price],
        key=lambda x: x[1]
    )[:num_levels]
    
    return supports, resistances

# =====================================================
# ЛИКВИДНОСТЬ
# =====================================================
def find_liquidity_levels(df, lookback=50):
    recent = df.tail(lookback)
    swing_highs = []
    swing_lows = []
    
    for i in range(2, len(recent) - 2):
        h = recent['high'].iloc[i]
        l = recent['low'].iloc[i]
        if (h > recent['high'].iloc[i-1] and h > recent['high'].iloc[i-2] and
            h > recent['high'].iloc[i+1] and h > recent['high'].iloc[i+2]):
            swing_highs.append(h)
        if (l < recent['low'].iloc[i-1] and l < recent['low'].iloc[i-2] and
            l < recent['low'].iloc[i+1] and l < recent['low'].iloc[i+2]):
            swing_lows.append(l)
    
    current_price = recent['close'].iloc[-1]
    liq_above = sorted([h for h in swing_highs if h > current_price])[:2]
    liq_below = sorted([l for l in swing_lows if l < current_price], reverse=True)[:2]
    
    return liq_above, liq_below

# =====================================================
# ПАТТЕРНЫ СВЕЧЕЙ
# =====================================================
def detect_candle_patterns(df):
    patterns = []
    last = df.iloc[-1]
    prev = df.iloc[-2]
    
    body = abs(last['close'] - last['open'])
    upper_shadow = last['high'] - max(last['close'], last['open'])
    lower_shadow = min(last['close'], last['open']) - last['low']
    total_range = last['high'] - last['low']
    
    if total_range == 0:
        return patterns, "NEUTRAL"
    
    if lower_shadow >= body * 2 and upper_shadow <= body * 0.3 and last['close'] > last['open']:
        patterns.append("🔨 Молот (бычий)")
    if upper_shadow >= body * 2 and lower_shadow <= body * 0.3 and last['close'] < last['open']:
        patterns.append("⭐ Падающая звезда (медвежий)")
    if body <= total_range * 0.1:
        patterns.append("➖ Доджи")
    if (last['close'] > last['open'] and prev['close'] < prev['open'] and
        last['open'] < prev['close'] and last['close'] > prev['open']):
        patterns.append("📈 Бычье поглощение")
    if (last['close'] < last['open'] and prev['close'] > prev['open'] and
        last['open'] > prev['close'] and last['close'] < prev['open']):
        patterns.append("📉 Медвежье поглощение")
    if lower_shadow >= total_range * 0.6 and body <= total_range * 0.3:
        patterns.append("📌 Пинбар (бычий)")
    if upper_shadow >= total_range * 0.6 and body <= total_range * 0.3:
        patterns.append("📌 Пинбар (медвежий)")
    
    bullish_count = sum(1 for p in patterns if any(w in p for w in ["бычий", "Молот", "Поглощение"]))
    bearish_count = sum(1 for p in patterns if any(w in p for w in ["медвежий", "Звезда"]))
    
    if bullish_count > bearish_count:
        pattern_signal = "BULLISH"
    elif bearish_count > bullish_count:
        pattern_signal = "BEARISH"
    else:
        pattern_signal = "NEUTRAL"
    
    return patterns, pattern_signal

# =====================================================
# ОБЪЁМ
# =====================================================
def analyze_volume(df):
    avg_volume = df['volume'].rolling(20).mean().iloc[-1]
    last_volume = df['volume'].iloc[-1]
    volume_ratio = last_volume / avg_volume if avg_volume > 0 else 1
    
    if volume_ratio >= 1.5:
        volume_signal = "STRONG"
        volume_emoji = "🔥"
    elif volume_ratio >= 1.2:
        volume_signal = "ABOVE"
        volume_emoji = "📊"
    else:
        volume_signal = "WEAK"
        volume_emoji = "⚠️"
    
    return {
        "volume_ratio": volume_ratio,
        "volume_signal": volume_signal,
        "volume_emoji": volume_emoji,
    }

# =====================================================
# СТРУКТУРНЫЙ АНАЛИЗ
# =====================================================
def detect_structure(df):
    last_10 = df.tail(10)
    highs = last_10['high'].values
    lows = last_10['low'].values
    current_price = df['close'].iloc[-1]
    prev_price = df['close'].iloc[-2]
    
    swing_high = max(highs[:-2])
    swing_low = min(lows[:-2])
    
    bos_bullish = current_price > swing_high and prev_price <= swing_high
    bos_bearish = current_price < swing_low and prev_price >= swing_low
    
    retest_bullish = (
        current_price > swing_high * 0.995 and
        current_price < swing_high * 1.005 and
        prev_price > swing_high
    )
    retest_bearish = (
        current_price < swing_low * 1.005 and
        current_price > swing_low * 0.995 and
        prev_price < swing_low
    )
    
    if bos_bullish or retest_bullish:
        return "BULLISH_STRUCTURE"
    elif bos_bearish or retest_bearish:
        return "BEARISH_STRUCTURE"
    else:
        return "NO_STRUCTURE"

# =====================================================
# АНАЛИЗ ОДНОГО ТАЙМФРЕЙМА
# =====================================================
def analyze_timeframe(df):
    df = df.copy()
    df['EMA20'] = df['close'].ewm(span=20).mean()
    df['EMA50'] = df['close'].ewm(span=50).mean()
    df['EMA200'] = df['close'].ewm(span=200).mean()
    df['RSI'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
    df['ATR'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
    df['ADX'] = ta.trend.ADXIndicator(df['high'], df['low'], df['close']).adx()
    
    last = df.iloc[-1]
    prev = df.iloc[-2]
    
    ema_diff_pct = abs(last['EMA20'] - last['EMA50']) / last['EMA50'] * 100
    if ema_diff_pct < 0.3:
        return "NO SIGNAL", last
    
    avg_atr = df['ATR'].rolling(20).mean().iloc[-1]
    if last['ATR'] < avg_atr * 0.8:
        return "NO SIGNAL", last
    
    price_move = abs(last['close'] - prev['close'])
    if price_move > last['ATR'] * 1.5:
        return "NO SIGNAL", last
    
    structure = detect_structure(df)
    
    long_trend = (last['close'] > last['EMA200'] and last['EMA20'] > last['EMA50'])
    long_rsi = 50 < last['RSI'] < 70
    long_structure = structure == "BULLISH_STRUCTURE"
    
    short_trend = (last['close'] < last['EMA200'] and last['EMA20'] < last['EMA50'])
    short_rsi = 30 < last['RSI'] < 50
    short_structure = structure == "BEARISH_STRUCTURE"
    
    if long_trend and long_rsi and long_structure:
        return "LONG", last
    elif short_trend and short_rsi and short_structure:
        return "SHORT", last
    else:
        return "NO SIGNAL", last

# =====================================================
# КОНТЕКСТ BTC
# =====================================================
btc_context_cache = {"value": "TRENDING", "timestamp": 0}

async def get_btc_context_cached():
    global btc_context_cache
    now = datetime.now().timestamp()
    
    if now - btc_context_cache["timestamp"] < 600:
        return btc_context_cache["value"]
    
    try:
        ohlcv = exchange.fetch_ohlcv("BTC/USDT", timeframe='1h', limit=50)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['EMA20'] = df['close'].ewm(span=20).mean()
        df['EMA50'] = df['close'].ewm(span=50).mean()
        last = df.iloc[-1]
        
        ema_diff_pct = abs(last['EMA20'] - last['EMA50']) / last['EMA50'] * 100
        
        if ema_diff_pct < 0.5:
            result = "FLAT"
        else:
            result = "TRENDING"
        
        btc_context_cache = {"value": result, "timestamp": now}
        logger.info(f"BTC контекст обновлён: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Ошибка BTC контекста: {e}")
        return "TRENDING"

# =====================================================
# АНАЛИЗ МОНЕТЫ
# =====================================================
def analyze_symbol(symbol, timeframes, btc_context):
    try:
        if symbol == "BTC/USDT":
            btc_context = "TRENDING"
        
        if btc_context == "FLAT":
            import random
            if random.random() < 0.5:
                return {"side": "NO SIGNAL", "btc_context": "FLAT"}
        
        results = {}
        for tf in timeframes:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe=tf, limit=250)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            side, last = analyze_timeframe(df)
            results[tf] = {"side": side, "last": last, "df": df}
        
        side_4h = results['4h']['side']
        last_4h = results['4h']['last']
        allowed_direction = "LONG" if last_4h['close'] > last_4h['EMA200'] else "SHORT"
        
        side_1d = results['1d']['side']
        side_1h = results['1h']['side']
        side_30m = results['30m']['side']
        side_15m = results['15m']['side']
        
        if side_1d == "NO SIGNAL" or side_4h == "NO SIGNAL":
            return {"side": "NO SIGNAL", "btc_context": btc_context}
        if side_1d != side_4h:
            return {"side": "NO SIGNAL", "btc_context": btc_context}
        if side_1h != side_1d:
            return {"side": "NO SIGNAL", "btc_context": btc_context}
        
        junior_confirms = (side_30m == side_1d) or (side_15m == side_1d)
        if not junior_confirms:
            return {"side": "NO SIGNAL", "btc_context": btc_context}
        
        final_side = side_1d
        
        if final_side != allowed_direction:
            return {"side": "NO SIGNAL", "btc_context": btc_context}
        
        df_1h = results['1h']['df']
        volume_data = analyze_volume(df_1h)
        if volume_data['volume_signal'] == "WEAK":
            return {"side": "NO SIGNAL", "btc_context": btc_context}
        
        df_15m = results['15m']['df']
        patterns, pattern_signal = detect_candle_patterns(df_15m)
        if final_side == "LONG" and pattern_signal == "BEARISH":
            return {"side": "NO SIGNAL", "btc_context": btc_context}
        if final_side == "SHORT" and pattern_signal == "BULLISH":
            return {"side": "NO SIGNAL", "btc_context": btc_context}
        
        supports, resistances = find_support_resistance(df_1h)
        liq_above, liq_below = find_liquidity_levels(df_1h)
        
        last_1h = results['1h']['last']
        trade_plan = build_advanced_trade_plan(last_1h['close'], last_1h['ATR'], final_side)
        
        return {
            "symbol": symbol,
            "side": final_side,
            "current_price": last_1h['close'],
            "rsi": last_1h['RSI'],
            "adx": last_1h['ADX'],
            "ema20": last_1h['EMA20'],
            "ema50": last_1h['EMA50'],
            "ema200": last_1h['EMA200'],
            "tf_1d": side_1d,
            "tf_4h": side_4h,
            "tf_1h": side_1h,
            "tf_30m": side_30m,
            "tf_15m": side_15m,
            "btc_context": btc_context,
            "volume_data": volume_data,
            "patterns": patterns,
            "supports": supports,
            "resistances": resistances,
            "liq_above": liq_above,
            "liq_below": liq_below,
            **trade_plan
        }
        
    except Exception as e:
        logger.error(f"Ошибка анализа {symbol}: {e}")
        return {"side": "NO SIGNAL", "btc_context": "ERROR"}

# =====================================================
# АСИНХРОННЫЙ ЗАПУСК
# =====================================================
async def analyze_all_timeframes_async(symbol, btc_context=None):
    if btc_context is None:
        btc_context = await get_btc_context_cached()
    
    loop = asyncio.get_running_loop()
    signal = await loop.run_in_executor(None, analyze_symbol, symbol, TIMEFRAMES, btc_context)
    return signal

# =====================================================
# ФОРМАТИРОВАНИЕ
# =====================================================
def tf_emoji(side):
    if side == "LONG":
        return "📈"
    elif side == "SHORT":
        return "📉"
    else:
        return "➖"

# =====================================================
# ФОРМАТИРОВАНИЕ СИГНАЛА
# =====================================================
def format_signal(signal_data):
    """Форматирует сигнал для отправки в Telegram"""
    if not signal_data or signal_data.get("side") == "NO SIGNAL":
        return None
    
    symbol = signal_data.get("symbol", "Unknown")
    side = signal_data.get("side", "UNKNOWN")
    price = signal_data.get("current_price", 0)
    
    # Эмодзи направления
    side_emoji = "📈" if side == "LONG" else "📉"
    
    # Форматируем уровни
    entry_min = signal_data.get("entry_min", 0)
    entry_max = signal_data.get("entry_max", 0)
    stop_loss = signal_data.get("stop_loss", 0)
    invalidation = signal_data.get("invalidation", 0)
    tp1 = signal_data.get("tp1", 0)
    tp2 = signal_data.get("tp2", 0)
    tp3 = signal_data.get("tp3", 0)
    
    # Индикаторы
    rsi = signal_data.get("rsi", 0)
    adx = signal_data.get("adx", 0)
    ema20 = signal_data.get("ema20", 0)
    ema50 = signal_data.get("ema50", 0)
    ema200 = signal_data.get("ema200", 0)
    
    # Таймфреймы
    tf_1d = signal_data.get("tf_1d", "")
    tf_4h = signal_data.get("tf_4h", "")
    tf_1h = signal_data.get("tf_1h", "")
    tf_30m = signal_data.get("tf_30m", "")
    tf_15m = signal_data.get("tf_15m", "")
    
    # BTC контекст
    btc_context = signal_data.get("btc_context", "UNKNOWN")
    btc_emoji = "🔥" if btc_context == "TRENDING" else "😴"
    
    # Объём
    volume_data = signal_data.get("volume_data", {})
    volume_emoji = volume_data.get("volume_emoji", "📊")
    volume_ratio = volume_data.get("volume_ratio", 0)
    
    # Паттерны
    patterns = signal_data.get("patterns", [])
    patterns_str = "\n".join(patterns) if patterns else "Нет паттернов"
    
    # Формируем сообщение
    message = f"""
{side_emoji} *{symbol}* - {side}

💰 *Цена:* ${price:.2f}
📊 *RSI:* {rsi:.1f} | *ADX:* {adx:.1f}
📈 *EMA20:* {ema20:.2f}
📈 *EMA50:* {ema50:.2f}
📈 *EMA200:* {ema200:.2f}

🎯 *Точка входа:* ${entry_min:.2f} - ${entry_max:.2f}
🛑 *Stop Loss:* ${stop_loss:.2f}
❌ *Invalidation:* ${invalidation:.2f}

📌 *Take Profit:*
  TP1: ${tp1:.2f}
  TP2: ${tp2:.2f}
  TP3: ${tp3:.2f}

📊 *Анализ таймфреймов:*
  1D: {tf_emoji(tf_1d)} {tf_1d}
  4H: {tf_emoji(tf_4h)} {tf_4h}
  1H: {tf_emoji(tf_1h)} {tf_1h}
  30m: {tf_emoji(tf_30m)} {tf_30m}
  15m: {tf_emoji(tf_15m)} {tf_15m}

🌍 *BTC контекст:* {btc_emoji} {btc_context}
📊 *Объём:* {volume_emoji} x{volume_ratio:.2f}

🔍 *Паттерны:*
{patterns_str}

⚠️ *Не забывайте про риск-менеджмент!*
"""
    
    return message