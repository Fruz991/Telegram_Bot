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
from macro_tracker import macro_tracker

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

    # ADX фильтр - отсеивает флэтовые сигналы
    if last['ADX'] < 20:
        logger.debug(f"ADX фильтр: ADX={last['ADX']:.2f} < 20 (флэт)")
        return "NO SIGNAL", last

    ema_diff_pct = abs(last['EMA20'] - last['EMA50']) / last['EMA50'] * 100
    if ema_diff_pct < 0.3:
        logger.debug(f"EMA фильтр: разница={ema_diff_pct:.3f}% < 0.3%")
        return "NO SIGNAL", last

    avg_atr = df['ATR'].rolling(20).mean().iloc[-1]
    if last['ATR'] < avg_atr * 0.8:
        logger.debug(f"ATR фильтр: ATR={last['ATR']:.2f} < {avg_atr * 0.8:.2f} (низкая волатильность)")
        return "NO SIGNAL", last

    price_move = abs(last['close'] - prev['close'])
    if price_move > last['ATR'] * 1.5:
        logger.debug(f"Price move фильтр: движение={price_move:.2f} > {last['ATR'] * 1.5:.2f} (резкий скачок)")
        return "NO SIGNAL", last

    # Проверка наклона EMA20
    ema20_prev = df['EMA20'].iloc[-2] if len(df) > 1 else last['EMA20']
    ema20_slope = (last['EMA20'] - ema20_prev) / ema20_prev * 100 if ema20_prev != 0 else 0

    structure = detect_structure(df)

    long_trend = (
        last['close'] > last['EMA200'] and 
        last['EMA20'] > last['EMA50'] and
        ema20_slope > 0  # EMA20 растёт
    )
    long_rsi = 50 < last['RSI'] < 70
    long_structure = structure == "BULLISH_STRUCTURE"

    short_trend = (
        last['close'] < last['EMA200'] and 
        last['EMA20'] < last['EMA50'] and
        ema20_slope < 0  # EMA20 падает
    )
    short_rsi = 30 < last['RSI'] < 50
    short_structure = structure == "BEARISH_STRUCTURE"

    if long_trend and long_rsi and long_structure:
        return "LONG", last
    elif short_trend and short_rsi and short_structure:
        return "SHORT", last
    else:
        return "NO SIGNAL", last

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
        logger.info(f"BTC контекст обновлён: {result}")
        return result

    except Exception as e:
        logger.error(f"Ошибка BTC контекста: {e}")
        return "FLAT"

async def get_market_context_cached():
    """Возвращает полный контекст: BTC + Макро (DXY + SPX)"""
    btc_context = await get_btc_context_cached()
    macro_context = macro_tracker.get_market_context()
    
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
# АНАЛИЗ МОНЕТЫ
# =====================================================
def analyze_symbol(symbol, timeframes, market_context):
    """
    Анализирует монету с учётом макро контекста

    market_context может быть:
    - строкой ("TRENDING"/"FLAT") для обратной совместимости
    - словарём {"btc": "...", "macro": {...}}
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
            logger.debug(f"{symbol}: 1D/4H NO SIGNAL")
            return {"side": "NO SIGNAL", "btc_context": btc_context}
        if side_1d != side_4h:
            logger.debug(f"{symbol}: 1D ({side_1d}) != 4H ({side_4h})")
            return {"side": "NO SIGNAL", "btc_context": btc_context}
        if side_1h != side_1d:
            logger.debug(f"{symbol}: 1H ({side_1h}) != 1D ({side_1d})")
            return {"side": "NO SIGNAL", "btc_context": btc_context}

        junior_confirms = (side_30m == side_1d) or (side_15m == side_1d)
        if not junior_confirms:
            logger.debug(f"{symbol}: Младшие TF не подтверждают (30m={side_30m}, 15m={side_15m})")
            return {"side": "NO SIGNAL", "btc_context": btc_context}

        final_side = side_1d

        if final_side != allowed_direction:
            logger.debug(f"{symbol}: Сигнал против направления 4H (allowed={allowed_direction})")
            return {"side": "NO SIGNAL", "btc_context": btc_context}

        # Проверка макро уклона — блокируем сигналы против макро
        if macro_bias != "NEUTRAL" and final_side != macro_bias:
            logger.info(f"{symbol}: БЛОК макро: signal={final_side}, macro_bias={macro_bias}")
            return {"side": "NO SIGNAL", "btc_context": btc_context}

        # Проверка BTC направления — блокируем сигналы против BTC (кроме FLAT)
        if btc_bias != "NEUTRAL" and final_side != btc_bias:
            logger.info(f"{symbol}: БЛОК BTC: signal={final_side}, btc_bias={btc_bias}")
            return {"side": "NO SIGNAL", "btc_context": btc_context}

        df_1h = results['1h']['df']
        volume_data = analyze_volume(df_1h)

        # Ужесточаем фильтр объёма при FLAT BTC
        if flat_mode:
            if volume_data['volume_signal'] not in ["STRONG", "ABOVE"] or volume_data['volume_ratio'] < 1.5:
                logger.debug(f"{symbol}: FLAT BTC + слабый объём (ratio={volume_data['volume_ratio']:.2f})")
                return {"side": "NO SIGNAL", "btc_context": btc_context}
        else:
            if volume_data['volume_signal'] == "WEAK":
                logger.debug(f"{symbol}: Объём WEAK (ratio={volume_data['volume_ratio']:.2f})")
                return {"side": "NO SIGNAL", "btc_context": btc_context}

        df_15m = results['15m']['df']
        patterns, pattern_signal = detect_candle_patterns(df_15m)
        if final_side == "LONG" and pattern_signal == "BEARISH":
            logger.debug(f"{symbol}: Паттерны против LONG: {patterns}")
            return {"side": "NO SIGNAL", "btc_context": btc_context}
        if final_side == "SHORT" and pattern_signal == "BULLISH":
            logger.debug(f"{symbol}: Паттерны против SHORT: {patterns}")
            return {"side": "NO SIGNAL", "btc_context": btc_context}

        # Проверка дивергенций RSI
        df_1h = results['1h']['df']
        divergence = detect_rsi_divergence(df_1h, window=5)

        # Дивергенция подтверждает сигнал - повышаем уверенность
        divergence_confirms = False
        if final_side == "LONG" and divergence == "BULLISH_DIVERGENCE":
            divergence_confirms = True
        if final_side == "SHORT" and divergence == "BEARISH_DIVERGENCE":
            divergence_confirms = True

        # При FLAT BTC требуем подтверждение дивергенцией
        if flat_mode and not divergence_confirms:
            logger.info(f"{symbol}: FLAT BTC без дивергенции (divergence={divergence})")
            return {"side": "NO SIGNAL", "btc_context": btc_context}

        supports, resistances = find_support_resistance(df_1h)
        liq_above, liq_below = find_liquidity_levels(df_1h)

        last_1h = results['1h']['last']
        trade_plan = build_advanced_trade_plan(last_1h['close'], last_1h['ATR'], final_side)

        # Формируем базовый сигнал
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
            **trade_plan
        }
        
        # Добавляем макро данные если есть
        if macro_context:
            signal["macro"] = macro_context

        logger.info(f"✅ СИГНАЛ: {symbol} {final_side} | BTC={btc_context} | ADX={last_1h['ADX']:.1f} | RSI={last_1h['RSI']:.1f}")
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
    message = f"""
{side_emoji} *{symbol}* - {side}

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