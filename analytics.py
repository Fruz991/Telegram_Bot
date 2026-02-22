import asyncio
import os
import socket
import logging
import time
import aiohttp
import ccxt
import pandas as pd
import numpy as np
import ta
import pytz
from datetime import datetime, timezone, timedelta
from config import TIMEFRAMES

logger = logging.getLogger(__name__)

# =====================================================
# IPv4 ТОЛЬКО ДЛЯ EXCHANGE
# =====================================================
def init_ipv4_only():
    """Инициализирует IPv4 только для ccxt exchange"""
    original_getaddrinfo = socket.getaddrinfo
    
    def getaddrinfo_ipv4(*args, **kwargs):
        results = original_getaddrinfo(*args, **kwargs)
        return [r for r in results if r[0].name == 'AF_INET']
    
    socket.getaddrinfo = getaddrinfo_ipv4
    logger.info("IPv4 only режим активирован")

init_ipv4_only()

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
# КЭШ ДЛЯ OHLCV ДАННЫХ
# =====================================================
ohlcv_cache = {}
ohlcv_cache_duration = 60  # 1 минута - достаточно для стабильности

def get_ohlcv_cached(symbol, timeframe, limit=250):
    """Кэширует запросы к бирже для экономии лимитов"""
    now = time.time()
    cache_key = f"{symbol}_{timeframe}"
    
    if cache_key in ohlcv_cache:
        cache_data = ohlcv_cache[cache_key]
        if now - cache_data["timestamp"] < ohlcv_cache_duration:
            return cache_data["data"]
    
    # Запрос к бирже
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    ohlcv_cache[cache_key] = {"data": ohlcv, "timestamp": now}
    return ohlcv

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
# ПАТТЕРНЫ СВЕЧЕЙ (НОВАЯ ВЕРСИЯ)
# =====================================================
def detect_candle_patterns(df):
    """
    Проверяет только сильное поглощение против тренда
    """
    patterns = []
    last = df.iloc[-1]
    prev = df.iloc[-2]

    body = abs(last['close'] - last['open'])
    prev_body = abs(prev['close'] - prev['open'])

    # Сильное бычье поглощение (тело последней свечи в 2+ раза больше предыдущей)
    if (last['close'] > last['open'] and prev['close'] < prev['open'] and
        last['open'] < prev['close'] and last['close'] > prev['open'] and
        body >= prev_body * 1.5):
        patterns.append("📈 Бычье поглощение")

    # Сильное медвежье поглощение
    if (last['close'] < last['open'] and prev['close'] > prev['open'] and
        last['open'] > prev['close'] and last['close'] < prev['open'] and
        body >= prev_body * 1.5):
        patterns.append("📉 Медвежье поглощение")

    # Определяем сигнал
    if len(patterns) > 0:
        if "Бычье поглощение" in str(patterns):
            pattern_signal = "BULLISH"
        elif "Медвежье поглощение" in str(patterns):
            pattern_signal = "BEARISH"
        else:
            pattern_signal = "NEUTRAL"
    else:
        pattern_signal = "NEUTRAL"  # Нет паттернов - нейтрально

    return patterns, pattern_signal

# =====================================================
# ОБЪЁМ (НОВАЯ ВЕРСИЯ)
# =====================================================
def analyze_volume(df):
    """
    Анализирует объем с новой логикой:
    - Объем > 90% от среднего ИЛИ
    - Текущий объем > предыдущего
    """
    avg_volume = df['volume'].rolling(20).mean().iloc[-1]
    last_volume = df['volume'].iloc[-1]
    prev_volume = df['volume'].iloc[-2] if len(df) > 1 else last_volume
    
    volume_ratio = last_volume / avg_volume if avg_volume > 0 else 1
    volume_vs_prev = last_volume / prev_volume if prev_volume > 0 else 1
    
    # Новый критерий: объем > 90% среднего ИЛИ > предыдущего
    volume_ok = volume_ratio >= 0.9 or volume_vs_prev >= 1.0
    
    if volume_ratio >= 1.5:
        volume_signal = "STRONG"
        volume_emoji = "🔥"
    elif volume_ratio >= 0.9:
        volume_signal = "ABOVE"
        volume_emoji = "📊"
    else:
        volume_signal = "WEAK"
        volume_emoji = "⚠️"

    return {
        "volume_ratio": volume_ratio,
        "volume_vs_prev": volume_vs_prev,
        "volume_signal": volume_signal,
        "volume_emoji": volume_emoji,
        "volume_ok": volume_ok
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
# ДИВЕРГЕНЦИИ RSI
# =====================================================
def detect_rsi_divergence(df, window=5):
    """
    Ищет бычьи/медвежьи дивергенции между ценой и RSI
    
    Бычья дивергенция: цена делает lower low, RSI делает higher low
    Медвежья дивергенция: цена делает higher high, RSI делает lower high
    """
    if len(df) < window * 2:
        return "NO_DIVERGENCE"
    
    last = df.iloc[-1]
    
    # Бычья дивергенция
    price_lows = df['low'].tail(window * 2).values
    rsi_lows = df['RSI'].tail(window * 2).values
    
    # Проверяем: цена делает lower low, RSI делает higher low
    if (price_lows[-1] < price_lows[0] and rsi_lows[-1] > rsi_lows[0]):
        logger.debug("🟢 Бычья дивергенция обнаружена")
        return "BULLISH_DIVERGENCE"
    
    # Медвежья дивергенция
    price_highs = df['high'].tail(window * 2).values
    rsi_highs = df['RSI'].tail(window * 2).values
    
    # Проверяем: цена делает higher high, RSI делает lower high
    if (price_highs[-1] > price_highs[0] and rsi_highs[-1] < rsi_highs[0]):
        logger.debug("🔴 Медвежья дивергенция обнаружена")
        return "BEARISH_DIVERGENCE"
    
    return "NO_DIVERGENCE"

# =====================================================
# АНАЛИЗ ОДНОГО ТАЙМФРЕЙМА (НОВАЯ ВЕРСИЯ)
# =====================================================
def analyze_timeframe(df, check_ema_cross=False):
    """
    Анализирует таймфрейм с новой логикой
    
    Args:
        df: DataFrame с OHLCV
        check_ema_cross: Если True - проверяем пересечение EMA за последние 5-8 свечей
    
    Returns:
        (side, last, score, details)
        side: "LONG"/"SHORT"/"NO SIGNAL"
        last: последние данные
        score: количество баллов (0-5)
        details: детали анализа
    """
    df = df.copy()
    df['EMA20'] = df['close'].ewm(span=20).mean()
    df['EMA50'] = df['close'].ewm(span=50).mean()
    df['EMA200'] = df['close'].ewm(span=200).mean()
    df['RSI'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
    df['ATR'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
    df['ADX'] = ta.trend.ADXIndicator(df['high'], df['low'], df['close']).adx()

    last = df.iloc[-1]
    prev = df.iloc[-2]
    
    details = {
        'adx': last['ADX'],
        'ema20': last['EMA20'],
        'ema50': last['EMA50'],
        'ema200': last['EMA200'],
        'atr': last['ATR'],
        'rsi': last['RSI'],
        'price': last['close']
    }

    # ===== 1. ADX фильтр (1 балл) =====
    adx_ok = last['ADX'] >= 20
    if not adx_ok:
        details['adx_ok'] = False

    # ===== 2. EMA анализ (1 балл + 1 бонус за пересечение) =====
    ema_diff_pct = abs(last['EMA20'] - last['EMA50']) / last['EMA50'] * 100
    ema20_above_ema50 = last['EMA20'] > last['EMA50']
    
    # Проверяем пересечение EMA за последние 5-8 свечей
    ema_cross_last_5 = False
    if check_ema_cross and len(df) >= 8:
        for i in range(5, 9):
            if len(df) >= i:
                prev_ema20 = df['EMA20'].iloc[-i]
                prev_ema50 = df['EMA50'].iloc[-i]
                # Было ниже, стало выше - пересечение вверх
                if prev_ema20 <= prev_ema50 and ema20_above_ema50:
                    ema_cross_last_5 = True
                    break
    
    ema_ok = ema_diff_pct >= 0.15  # Разница > 0.15%
    details['ema_diff_pct'] = ema_diff_pct
    details['ema_cross'] = ema_cross_last_5

    # ===== 3. ATR фильтр (1 балл) =====
    avg_atr = df['ATR'].rolling(20).mean().iloc[-1]
    atr_ok = last['ATR'] >= avg_atr * 0.65  # ATR > 65% от среднего
    details['atr_ok'] = atr_ok
    details['atr_ratio'] = last['ATR'] / avg_atr if avg_atr > 0 else 0

    # ===== 4. Цена vs EMA200 (2 балла) =====
    price_above_ema200 = last['close'] > last['EMA200']
    
    # Проверяем отскок от EMA200 (цена была ниже, коснулась, стала выше)
    bounce_off_ema200 = False
    if len(df) >= 3:
        for i in range(1, 4):
            prev_close = df['close'].iloc[-i]
            prev_ema200 = df['EMA200'].iloc[-i]
            # Была ниже или на уровне, теперь выше
            if prev_close <= prev_ema200 * 1.002 and price_above_ema200:
                bounce_off_ema200 = True
                break
    
    ema200_ok = price_above_ema200 or bounce_off_ema200
    ema200_score = 2 if ema200_ok else 0
    details['price_above_ema200'] = price_above_ema200
    details['bounce_off_ema200'] = bounce_off_ema200

    # ===== 5. Структурный анализ =====
    structure = detect_structure(df)
    details['structure'] = structure

    # ===== Определяем направление =====
    long_trend = ema20_above_ema50
    short_trend = last['EMA20'] < last['EMA50']

    # ===== Подсчет баллов =====
    score = 0
    if adx_ok:
        score += 1
    if ema_ok or ema_cross_last_5:
        score += 1
    if ema_cross_last_5:
        score += 1  # Бонус за пересечение
    if atr_ok:
        score += 1
    score += ema200_score  # 0 или 2 балла

    # ===== Итоговое решение =====
    if long_trend and score >= 4:
        return "LONG", last, score, details
    elif short_trend and score >= 4:
        return "SHORT", last, score, details
    else:
        return "NO SIGNAL", last, score, details

# =====================================================
# КОНТЕКСТ BTC + МАКРО
# =====================================================
btc_context_cache = {"value": "FLAT", "timestamp": 0}

async def get_btc_context_cached():
    """
    Возвращает контекст BTC с 3 состояниями:
    - BULL: Бычий тренд (цена > EMA200, EMA20 > EMA50)
    - BEAR: Медвежий тренд (цена < EMA200, EMA20 < EMA50)
    - FLAT: Боковик (EMA20 и EMA50 близко)
    """
    global btc_context_cache
    now = datetime.now().timestamp()

    if now - btc_context_cache["timestamp"] < 300:  # 5 минут
        return btc_context_cache["value"]

    try:
        ohlcv = get_ohlcv_cached("BTC/USDT", timeframe='1h', limit=50)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['EMA20'] = df['close'].ewm(span=20).mean()
        df['EMA50'] = df['close'].ewm(span=50).mean()
        df['EMA200'] = df['close'].ewm(span=200).mean()
        last = df.iloc[-1]

        ema_diff_pct = abs(last['EMA20'] - last['EMA50']) / last['EMA50'] * 100
        price = last['close']
        ema200 = last['EMA200']

        # Сначала проверяем на FLAT (приоритет)
        if ema_diff_pct < 0.8:  # Смягчил с 0.5% до 0.8%
            result = "FLAT"
        # Затем определяем направление тренда
        elif price > ema200 and last['EMA20'] > last['EMA50']:
            result = "BULL"
        elif price < ema200 and last['EMA20'] < last['EMA50']:
            result = "BEAR"
        else:
            result = "FLAT"  # Если сигналы противоречивы

        btc_context_cache = {"value": result, "timestamp": now}
        return result

    except Exception as e:
        logger.error(f"Ошибка BTC контекста: {e}")
        return "FLAT"

async def get_market_context_cached():
    """Возвращает полный контекст: BTC + Макро (DXY + SPX)"""
    btc_context = await get_btc_context_cached()
    # Макро отключено временно
    macro_context = None

    return {
        "btc": btc_context,
        "macro": macro_context
    }

# =====================================================
# МАКРО НАПРАВЛЕНИЕ РЫНКА
# =====================================================
def get_macro_bias(macro_context):
    """
    Определяет макро уклон рынка на основе DXY и SPX
    
    Логика (вспомогательный фильтр — только сильные тренды):
    - DXY сильно растёт (STRONG) + SPX сильно падает (WEAK) = медвежий уклон 📉
    - DXY сильно падает (WEAK) + SPX сильно растёт (STRONG) = бычий уклон 📈
    - Иначе = нейтрально (не блокируем сигналы)
    
    Возвращает: "LONG", "SHORT", "NEUTRAL"
    """
    if not macro_context:
        return "NEUTRAL"
    
    spx = macro_context.get("spx", {})
    dxy = macro_context.get("dxy", {})
    
    spx_trend = spx.get("trend", "FLAT")
    dxy_trend = dxy.get("trend", "FLAT")
    
    # Проверяем только сильные тренды по обоим индикаторам
    # Бычий сценарий: SPX растёт (STRONG) + DXY падает (WEAK)
    if spx_trend == "STRONG" and dxy_trend == "WEAK":
        return "LONG"
    
    # Медвежий сценарий: SPX падает (WEAK) + DXY растёт (STRONG)
    if spx_trend == "WEAK" and dxy_trend == "STRONG":
        return "SHORT"
    
    # Все остальные случаи — нейтрально (не блокируем)
    return "NEUTRAL"

# =====================================================
# АНАЛИЗ МОНЕТЫ (НОВАЯ ВЕРСИЯ С БАЛЛАМИ)
# =====================================================
def analyze_symbol(symbol, timeframes, market_context):
    """
    Анализирует монету с новой системой баллов
    
    Балльная система:
    - 4H и 1H совпадают (направление) = 2 балла
    - ADX > 20 = 1 балл
    - EMA20 > EMA50 (или пересечение) = 1 балл (+1 бонус за пересечение)
    - ATR > 65% от среднего = 1 балл
    - Объем > 90% среднего = 1 балл
    - Цена выше EMA200 (или отскок) = 2 балла
    - 1D подтверждает = +1 бонус
    
    Максимум: 8 баллов
    Минимум для входа: 6 баллов
    """
    # Обрабатываем старый формат (для совместимости)
    if isinstance(market_context, str):
        btc_context = market_context
        macro_context = None
    else:
        btc_context = market_context.get("btc", "TRENDING")
        macro_context = market_context.get("macro", None)

    try:
        if symbol == "BTC/USDT":
            btc_context = "BULL"  # Для BTC не блокируем по направлению

        # Определяем режим по BTC контексту
        flat_mode = btc_context == "FLAT"
        btc_bias = "LONG" if btc_context == "BULL" else ("SHORT" if btc_context == "BEAR" else "NEUTRAL")

        # Получаем макро уклон
        macro_bias = get_macro_bias(macro_context)

        results = {}
        for tf in timeframes:
            ohlcv = get_ohlcv_cached(symbol, timeframe=tf, limit=250)
            if not ohlcv or len(ohlcv) < 50:
                logger.warning(f"Недостаточно данных для {symbol} {tf}")
                return {"side": "NO SIGNAL", "btc_context": btc_context}

            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            # Для 4H и 1H проверяем пересечение EMA
            check_cross = tf in ['4h', '1h']
            side, last, score, details = analyze_timeframe(df, check_ema_cross=check_cross)
            results[tf] = {"side": side, "last": last, "score": score, "details": details, "df": df}

        # ===== ГЛАВНОЕ УСЛОВИЕ: 4H и 1H должны совпадать =====
        side_4h = results['4h']['side']
        side_1h = results['1h']['side']
        
        # Если 4H или 1H не имеют сигнала - нет общего сигнала
        if side_4h == "NO SIGNAL" or side_1h == "NO SIGNAL":
            return {"side": "NO SIGNAL", "btc_context": btc_context}
        
        # Если 4H и 1H не совпадают - нет сигнала
        if side_4h != side_1h:
            return {"side": "NO SIGNAL", "btc_context": btc_context}
        
        final_side = side_4h  # Направление по 4H и 1H

        # ===== 1D - не обязателен, но может усилить сигнал =====
        side_1d = results['1d']['side']
        day_confirms = 1 if side_1d == final_side else 0

        # ===== 30m и 15m - точка входа =====
        side_30m = results['30m']['side']
        side_15m = results['15m']['side']
        
        # Логика: 30m совпадает ИЛИ 15m совпадает ИЛИ (30m нейтрально но 15m дает импульс)
        entry_confirms = False
        if side_30m == final_side:
            entry_confirms = True  # 30m подтверждает
        elif side_15m == final_side:
            entry_confirms = True  # 15m подтверждает
        elif side_30m == "NO SIGNAL" and side_15m != "NO SIGNAL":
            # 30m нейтрально, но 15m дает импульс в любом направлении
            if side_15m == final_side:
                entry_confirms = True
        
        if not entry_confirms:
            return {"side": "NO SIGNAL", "btc_context": btc_context}

        # ===== Проверка направления по EMA200 на 4H =====
        last_4h = results['4h']['last']
        details_4h = results['4h']['details']
        
        # Цена выше EMA200 ИЛИ был отскок
        price_above_ema200 = details_4h.get('price_above_ema200', False)
        bounce_off_ema200 = details_4h.get('bounce_off_ema200', False)
        
        if not (price_above_ema200 or bounce_off_ema200):
            return {"side": "NO SIGNAL", "btc_context": btc_context}

        # ===== Проверка макро уклона =====
        if macro_bias != "NEUTRAL" and final_side != macro_bias:
            logger.info(f"🚫 {symbol}: БЛОК макро (signal={final_side}, macro_bias={macro_bias})")
            return {"side": "NO SIGNAL", "btc_context": btc_context}

        # ===== Проверка BTC направления =====
        if btc_bias != "NEUTRAL" and final_side != btc_bias:
            logger.info(f"🚫 {symbol}: БЛОК BTC (signal={final_side}, btc_bias={btc_bias})")
            return {"side": "NO SIGNAL", "btc_context": btc_context}

        # ===== Объём =====
        df_1h = results['1h']['df']
        volume_data = analyze_volume(df_1h)
        
        if not volume_data['volume_ok']:
            return {"side": "NO SIGNAL", "btc_context": btc_context}

        # ===== Свечные паттерны (только сильное поглощение против) =====
        df_15m = results['15m']['df']
        patterns, pattern_signal = detect_candle_patterns(df_15m)
        
        if final_side == "LONG" and pattern_signal == "BEARISH":
            return {"side": "NO SIGNAL", "btc_context": btc_context}
        if final_side == "SHORT" and pattern_signal == "BULLISH":
            return {"side": "NO SIGNAL", "btc_context": btc_context}

        # ===== FLAT BTC режим - альт должен иметь ADX > 23 =====
        if flat_mode and symbol != "BTC/USDT":
            adx_1h = results['1h']['details'].get('adx', 0)
            if adx_1h < 23:
                logger.info(f"🚫 {symbol}: FLAT BTC + ADX={adx_1h:.1f} < 23")
                return {"side": "NO SIGNAL", "btc_context": btc_context}

        # ===== Подсчет общих баллов =====
        total_score = 0
        
        # 4H и 1H совпадают = 2 балла
        total_score += 2
        
        # ADX > 20 на 1H = 1 балл
        if results['1h']['details'].get('adx', 0) >= 20:
            total_score += 1
        
        # EMA20 > EMA50 на 1H = 1 балл
        if results['1h']['details'].get('ema_diff_pct', 0) >= 0.15:
            total_score += 1
        elif results['1h']['details'].get('ema_cross', False):
            total_score += 2  # Бонус за пересечение
        
        # ATR > 65% на 1H = 1 балл
        if results['1h']['details'].get('atr_ratio', 0) >= 0.65:
            total_score += 1
        
        # Объем > 90% = 1 балл
        if volume_data.get('volume_ratio', 0) >= 0.9:
            total_score += 1
        
        # Цена выше EMA200 на 4H = 2 балла
        if price_above_ema200 or bounce_off_ema200:
            total_score += 2
        
        # 1D подтверждает = +1 бонус
        total_score += day_confirms

        # ===== Минимальный порог для сигнала =====
        if total_score < 6:
            logger.debug(f"{symbol}: Недостаточно баллов ({total_score}/6)")
            return {"side": "NO SIGNAL", "btc_context": btc_context}

        # ===== Дивергенции RSI (опционально) =====
        divergence = detect_rsi_divergence(df_1h, window=5)
        divergence_confirms = False
        if final_side == "LONG" and divergence == "BULLISH_DIVERGENCE":
            divergence_confirms = True
        if final_side == "SHORT" and divergence == "BEARISH_DIVERGENCE":
            divergence_confirms = True

        supports, resistances = find_support_resistance(df_1h)
        liq_above, liq_below = find_liquidity_levels(df_1h)

        last_1h = results['1h']['last']
        trade_plan = build_advanced_trade_plan(last_1h['close'], last_1h['ATR'], final_side)

        # Формируем сигнал
        signal = {
            "symbol": symbol,
            "side": final_side,
            "current_price": last_1h['close'],
            "rsi": last_1h['RSI'],
            "adx": last_1h['ADX'],
            "ema20": last_1h['EMA20'],
            "ema50": last_1h['EMA50'],
            "ema200": last_1h['EMA200'],
            "atr": last_1h['ATR'],
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
            "divergence": divergence,
            "divergence_confirms": divergence_confirms,
            "score": total_score,  # Добавляем счет
            **trade_plan
        }

        if macro_context:
            signal["macro"] = macro_context

        return signal

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
    atr = signal_data.get("atr", 0)

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
    
    # ATR в процентах для волатильности
    atr_percent = (atr / price * 100) if price > 0 and atr else 0

    # Таймфреймы
    tf_1d = signal_data.get("tf_1d", "")
    tf_4h = signal_data.get("tf_4h", "")
    tf_1h = signal_data.get("tf_1h", "")
    tf_30m = signal_data.get("tf_30m", "")
    tf_15m = signal_data.get("tf_15m", "")

    # BTC контекст
    btc_context = signal_data.get("btc_context", "UNKNOWN")
    btc_emoji = {"BULL": "🐂", "BEAR": "🐻", "FLAT": "😴"}

    # Объём
    volume_data = signal_data.get("volume_data", {})
    volume_emoji = volume_data.get("volume_emoji", "📊")
    volume_ratio = volume_data.get("volume_ratio", 0)

    # Паттерны
    patterns = signal_data.get("patterns", [])
    patterns_str = "\n".join(patterns) if patterns else "Нет паттернов"
    
    # Дивергенции
    divergence = signal_data.get("divergence", "NO_DIVERGENCE")
    divergence_confirms = signal_data.get("divergence_confirms", False)
    if divergence == "BULLISH_DIVERGENCE":
        divergence_str = "🟢 Бычья дивергенция"
    elif divergence == "BEARISH_DIVERGENCE":
        divergence_str = "🔴 Медвежья дивергенция"
    else:
        divergence_str = "➖ Нет дивергенций"
    
    if divergence_confirms:
        divergence_str += " ✅"
    
    # Расчёт RR для TP1
    if side == "LONG":
        rr_tp1 = (tp1 - price) / (price - stop_loss) if price != stop_loss else 0
    else:
        rr_tp1 = (price - tp1) / (stop_loss - price) if stop_loss != price else 0
    
    # Макро данные
    macro = signal_data.get("macro", {})
    spx = macro.get("spx", {})
    dxy = macro.get("dxy", {})
    crypto_impact = macro.get("crypto_impact", "NEUTRAL")
    
    # Макро уклон
    macro_bias = get_macro_bias(macro) if macro else "NEUTRAL"
    bias_emoji = {"LONG": "📈", "SHORT": "📉", "NEUTRAL": "➖"}
    bias_str = {
        "LONG": "Бычий (SPX↑ + DXY↓)",
        "SHORT": "Медвежий (SPX↓ + DXY↑)",
        "NEUTRAL": "Нейтральный"
    }

    spx_emoji = {"STRONG": "🟢", "WEAK": "🔴", "FLAT": "😐", "ERROR": "⚠️"}
    dxy_emoji = {"STRONG": "🔴", "WEAK": "🟢", "FLAT": "😐", "ERROR": "⚠️"}
    
    spx_str = "Нет данных"
    dxy_str = "Нет данных"
    
    if spx.get("value"):
        spx_change = spx.get("change", 0)
        spx_str = f"{spx_emoji.get(spx.get('trend', 'FLAT'), '😐')} {spx['value']:.2f} ({spx_change:+.2f}%)"
    
    if dxy.get("value"):
        dxy_change = dxy.get("change", 0)
        dxy_str = f"{dxy_emoji.get(dxy.get('trend', 'FLAT'), '😐')} {dxy['value']:.2f} ({dxy_change:+.2f}%)"
    
    impact_emoji = {"BULLISH": "🟢", "BEARISH": "🔴", "NEUTRAL": "😐"}
    impact_str = {
        "BULLISH": "Благоприятно для крипты",
        "BEARISH": "Давление на крипту",
        "NEUTRAL": "Нейтрально"
    }

    # Формируем сообщение
    score = signal_data.get("score", 0)
    
    message = f"""
{side_emoji} *{symbol}* - {side}
⭐ *Баллы:* {score}/8

💰 *Цена:* ${price:.2f}
📊 *RSI:* {rsi:.1f} | *ADX:* {adx:.1f}
📈 *EMA20:* {ema20:.2f}
📈 *EMA50:* {ema50:.2f}
📈 *EMA200:* {ema200:.2f}
📉 *ATR:* {atr:.4f} ({atr_percent:.2f}%)

🎯 *Точка входа:* ${entry_min:.2f} - ${entry_max:.2f}
🛑 *Stop Loss:* ${stop_loss:.2f}
❌ *Invalidation:* ${invalidation:.2f}

📌 *Take Profit:*
  TP1: ${tp1:.2f} (RR {rr_tp1:.2f})
  TP2: ${tp2:.2f}
  TP3: ${tp3:.2f}

📊 *Анализ таймфреймов:*
  1D: {tf_emoji(tf_1d)} {tf_1d}
  4H: {tf_emoji(tf_4h)} {tf_4h}
  1H: {tf_emoji(tf_1h)} {tf_1h}
  30m: {tf_emoji(tf_30m)} {tf_30m}
  15m: {tf_emoji(tf_15m)} {tf_15m}

🌍 *BTC контекст:* {btc_emoji.get(btc_context, '😴')} {btc_context}
📊 *Макро уклон:* {bias_emoji.get(macro_bias, '➖')} {bias_str.get(macro_bias, 'Нейтральный')}
🏛 *S&P 500:* {spx_str}
💵 *DXY:* {dxy_str}
🔀 *Влияние:* {impact_emoji.get(crypto_impact, '😐')} {impact_str.get(crypto_impact, 'Нейтрально')}
📊 *Объём:* {volume_emoji} x{volume_ratio:.2f}
🔀 *Дивергенции:* {divergence_str}

🔍 *Паттерны:*
{patterns_str}

⚠️ *Не забывайте про риск-менеджмент!*
"""

    return message