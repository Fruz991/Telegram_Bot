import asyncio
import os
import socket
import ccxt
import pandas as pd
import numpy as np
import ta
from dotenv import load_dotenv

from config import TIMEFRAMES

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
load_dotenv()

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

    for i, (type1, price1) in enumerate(levels):
        if i in used:
            continue
        cluster = [price1]
        for j, (type2, price2) in enumerate(levels):
            if j != i and j not in used and abs(price1 - price2) < price_range:
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
# СТРУКТУРНЫЙ АНАЛИЗ (BOS + РЕТЕСТ)
# =====================================================
def detect_structure(df):
    """
    Определяет Break of Structure и ретест
    BOS = цена пробивает последний swing high/low
    Ретест = возврат к пробитому уровню
    """
    last_10 = df.tail(10)
    
    highs = last_10['high'].values
    lows = last_10['low'].values
    
    current_price = df['close'].iloc[-1]
    prev_price = df['close'].iloc[-2]
    
    # Находим последний swing high и low
    swing_high = max(highs[:-2])  # Исключаем последние 2 свечи
    swing_low = min(lows[:-2])
    
    # BOS вверх (бычий)
    bos_bullish = current_price > swing_high and prev_price <= swing_high
    
    # BOS вниз (медвежий)
    bos_bearish = current_price < swing_low and prev_price >= swing_low
    
    # Ретест (цена вернулась к пробитому уровню)
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
    """
    Система: EMA + RSI + Объём + Структура
    """
    
    # --- EMA 20/50/200 ---
    df['EMA20']  = df['close'].ewm(span=20).mean()
    df['EMA50']  = df['close'].ewm(span=50).mean()
    df['EMA200'] = df['close'].ewm(span=200).mean()
    
    # --- RSI 14 ---
    df['RSI'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
    
    # --- ATR ---
    df['ATR'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
    
    # --- ADX ---
    df['ADX'] = ta.trend.ADXIndicator(df['high'], df['low'], df['close']).adx()
    
    last = df.iloc[-1]
    prev = df.iloc[-2]
    
    # ══════════════════════════════════════
    # ФИЛЬТР ФЛЭТА
    # ══════════════════════════════════════
    ema_diff_pct = abs(last['EMA20'] - last['EMA50']) / last['EMA50'] * 100
    if ema_diff_pct < 0.3:  # Флэт
        return "NO SIGNAL", last
    
    # ══════════════════════════════════════
    # ФИЛЬТР ВОЛАТИЛЬНОСТИ
    # Если ATR ниже среднего за 20 свечей — рынок пилит
    # ══════════════════════════════════════
    avg_atr = df['ATR'].rolling(20).mean().iloc[-1]
    if last['ATR'] < avg_atr * 0.8:  # Низкая волатильность
        return "NO SIGNAL", last
    
    # ══════════════════════════════════════
    # ФИЛЬТР СИЛЬНОГО ДВИЖЕНИЯ
    # ══════════════════════════════════════
    price_move = abs(last['close'] - prev['close'])
    if price_move > last['ATR'] * 1.5:
        return "NO SIGNAL", last
    
    # ══════════════════════════════════════
    # СТРУКТУРНЫЙ ФИЛЬТР
    # ══════════════════════════════════════
    structure = detect_structure(df)
    
    # ══════════════════════════════════════
    # LONG CONDITIONS
    # ══════════════════════════════════════
    long_trend = (
        last['close'] > last['EMA200'] and
        last['EMA20'] > last['EMA50']
    )
    long_rsi = 50 < last['RSI'] < 70
    long_structure = structure == "BULLISH_STRUCTURE"
    
    # ══════════════════════════════════════
    # SHORT CONDITIONS
    # ══════════════════════════════════════
    short_trend = (
        last['close'] < last['EMA200'] and
        last['EMA20'] < last['EMA50']
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
# КОНТЕКСТ BTC
# =====================================================
def get_btc_context():
    """
    Проверяет BTC на флэт
    Если BTC во флэте — снижаем частоту сигналов
    """
    try:
        ohlcv = exchange.fetch_ohlcv("BTC/USDT", timeframe='1h', limit=50)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        df['EMA20'] = df['close'].ewm(span=20).mean()
        df['EMA50'] = df['close'].ewm(span=50).mean()
        
        last = df.iloc[-1]
        
        ema_diff_pct = abs(last['EMA20'] - last['EMA50']) / last['EMA50'] * 100
        
        if ema_diff_pct < 0.5:
            return "FLAT"  # BTC во флэте
        else:
            return "TRENDING"  # BTC трендует
            
    except Exception as e:
        print(f"Ошибка BTC контекста: {e}")
        return "TRENDING"  # По умолчанию разрешаем


# =====================================================
# АНАЛИЗ МОНЕТЫ
# Система: 1D → 4H → 1H → 30m/15m
# =====================================================
def analyze_symbol(symbol, timeframes):
    try:
        # Пропускаем BTC при анализе альтов
        if symbol == "BTC/USDT":
            btc_context = "TRENDING"
        else:
            btc_context = get_btc_context()
            
            # Если BTC во флэте — пропускаем 50% сигналов
            if btc_context == "FLAT":
                import random
                if random.random() < 0.5:
                    return {"side": "NO SIGNAL", "btc_context": "FLAT"}
        
        results = {}
        
        for tf in timeframes:
            limit = 250
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe=tf, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            side, last = analyze_timeframe(df)
            results[tf] = {"side": side, "last": last, "df": df}
        
        # ══════════════════════════════════════
        # ФИЛЬТР СТАРШЕГО ТРЕНДА 4H
        # ══════════════════════════════════════
        side_4h = results['4h']['side']
        last_4h = results['4h']['last']
        
        # Если 4H выше EMA200 → только лонги, ниже → только шорты
        if last_4h['close'] > last_4h['EMA200']:
            allowed_direction = "LONG"
        else:
            allowed_direction = "SHORT"
        
        # ══════════════════════════════════════
        # СТАРШИЙ ПОДТВЕРЖДАЕТ МЛАДШЕГО
        # 1D → 4H → 1H обязательны
        # 30m или 15m подтверждает
        # ══════════════════════════════════════
        side_1d  = results['1d']['side']
        side_1h  = results['1h']['side']
        side_30m = results['30m']['side']
        side_15m = results['15m']['side']
        
        # 1D и 4H должны совпадать
        if side_1d == "NO SIGNAL" or side_4h == "NO SIGNAL":
            return {"side": "NO SIGNAL", "btc_context": btc_context}
        
        if side_1d != side_4h:
            return {"side": "NO SIGNAL", "btc_context": btc_context}
        
        # 1H подтверждает
        if side_1h != side_1d:
            return {"side": "NO SIGNAL", "btc_context": btc_context}
        
        # Хотя бы 30m или 15m подтверждает
        junior_confirms = (side_30m == side_1d) or (side_15m == side_1d)
        if not junior_confirms:
            return {"side": "NO SIGNAL", "btc_context": btc_context}
        
        final_side = side_1d
        
        # Проверка против старшего тренда 4H
        if final_side != allowed_direction:
            return {"side": "NO SIGNAL", "btc_context": btc_context}
        
        # ══════════════════════════════════════
        # ОБЪЁМ (1H)
        # ══════════════════════════════════════
        df_1h = results['1h']['df']
        volume_data = analyze_volume(df_1h)
        
        if volume_data['volume_signal'] == "WEAK":
            return {"side": "NO SIGNAL", "btc_context": btc_context}
        
        # ══════════════════════════════════════
        # ПАТТЕРНЫ (15M)
        # ══════════════════════════════════════
        df_15m = results['15m']['df']
        patterns, pattern_signal = detect_candle_patterns(df_15m)
        
        if final_side == "LONG" and pattern_signal == "BEARISH":
            return {"side": "NO SIGNAL", "btc_context": btc_context}
        if final_side == "SHORT" and pattern_signal == "BULLISH":
            return {"side": "NO SIGNAL", "btc_context": btc_context}
        
        # ══════════════════════════════════════
        # УРОВНИ (1H)
        # ══════════════════════════════════════
        supports, resistances = find_support_resistance(df_1h)
        liq_above, liq_below  = find_liquidity_levels(df_1h)
        
        # ══════════════════════════════════════
        # ТОРГОВЫЙ ПЛАН (1H)
        # ══════════════════════════════════════
        last_1h = results['1h']['last']
        trade_plan = build_advanced_trade_plan(
            last_1h['close'], last_1h['ATR'], final_side
        )
        
        return {
            "symbol":        symbol,
            "side":          final_side,
            "current_price": last_1h['close'],
            "rsi":           last_1h['RSI'],
            "adx":           last_1h['ADX'],
            "ema20":         last_1h['EMA20'],
            "ema50":         last_1h['EMA50'],
            "ema200":        last_1h['EMA200'],
            "tf_1d":         side_1d,
            "tf_4h":         side_4h,
            "tf_1h":         side_1h,
            "tf_30m":        side_30m,
            "tf_15m":        side_15m,
            "btc_context":   btc_context,
            "volume_data":   volume_data,
            "patterns":      patterns,
            "supports":      supports,
            "resistances":   resistances,
            "liq_above":     liq_above,
            "liq_below":     liq_below,
            **trade_plan
        }
        
    except Exception as e:
        print(f"Ошибка анализа {symbol}: {e}")
        return {"side": "NO SIGNAL", "btc_context": "ERROR"}


# =====================================================
# АСИНХРОННЫЙ ЗАПУСК
# =====================================================
async def analyze_all_timeframes_async(symbol):
    loop = asyncio.get_running_loop()
    signal = await loop.run_in_executor(None, analyze_symbol, symbol, TIMEFRAMES)
    return signal


# =====================================================
# ФОРМАТИРОВАНИЕ
# =====================================================
def tf_emoji(side):
    if side == "LONG":   return "📈"
    if side == "SHORT":  return "📉"
    return "⬜"


def format_signal(signal):
    symbol_fmt = signal['symbol'].replace('/', '')
    side       = signal['side']
    emoji      = "📈" if side == "LONG" else "📉"
    vol        = signal['volume_data']
    
    btc_status = "🟢 Трендует" if signal['btc_context'] == "TRENDING" else "🔴 Во флэте"
    
    patterns_text = "\n   ".join(signal['patterns']) if signal['patterns'] else "—"
    supports_text = " | ".join([f"{p:.4f}" for _, p in signal['supports']]) if signal['supports'] else "—"
    resistances_text = " | ".join([f"{p:.4f}" for _, p in signal['resistances']]) if signal['resistances'] else "—"
    liq_above_text = " | ".join([f"{p:.4f}" for p in signal['liq_above']]) if signal['liq_above'] else "—"
    liq_below_text = " | ".join([f"{p:.4f}" for p in signal['liq_below']]) if signal['liq_below'] else "—"
    
    return f"""
🚨 TRADE PLAN | {symbol_fmt} | {side}
{emoji} Подтверждение таймфреймов:
   1D: {tf_emoji(signal['tf_1d'])}  4H: {tf_emoji(signal['tf_4h'])}  1H: {tf_emoji(signal['tf_1h'])}  30M: {tf_emoji(signal['tf_30m'])}  15M: {tf_emoji(signal['tf_15m'])}

₿ BTC: {btc_status}

💰 Зона набора:
   {signal['entry_min']:.4f} — {signal['entry_max']:.4f}

🛑 Стоп-лосс:
   {signal['stop_loss']:.4f}

❌ Отмена идеи:
   H1 close {'<' if side == 'LONG' else '>'} {signal['invalidation']:.4f}

🎯 Тейки:
   TP1: {signal['tp1']:.4f}  (25%)
   TP2: {signal['tp2']:.4f}  (50%) — RR 1:2
   TP3: {signal['tp3']:.4f}  (25%) — RR 1:3

📊 Индикаторы:
   RSI: {signal['rsi']:.1f}   ADX: {signal['adx']:.1f}
   EMA20: {signal['ema20']:.4f}
   EMA50: {signal['ema50']:.4f}
   EMA200: {signal['ema200']:.4f}

📦 Объём: {vol['volume_emoji']} x{vol['volume_ratio']:.1f}

🕯 Паттерны:
   {patterns_text}

🏛 Поддержки: {supports_text}
🏛 Сопротивления: {resistances_text}

💧 Ликвидность выше: {liq_above_text}
💧 Ликвидность ниже: {liq_below_text}

💵 Цена: {signal['current_price']:.4f}
"""