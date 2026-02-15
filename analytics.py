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
# ТОРГОВЫЙ ПЛАН С RR 1:2 И 1:3
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
# УРОВНИ ПОДДЕРЖКИ И СОПРОТИВЛЕНИЯ
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
        patterns.append("➖ Доджи (неопределённость)")
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
# АНАЛИЗ ОБЪЁМА
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
    elif volume_ratio >= 0.7:
        volume_signal = "NORMAL"
        volume_emoji = "➡️"
    else:
        volume_signal = "WEAK"
        volume_emoji = "⚠️"

    return {
        "volume_ratio": volume_ratio,
        "volume_signal": volume_signal,
        "volume_emoji": volume_emoji,
    }


# =====================================================
# АНАЛИЗ ОДНОГО ТАЙМФРЕЙМА
# Возвращает: "LONG" / "SHORT" / "NO SIGNAL" + данные
# =====================================================
def analyze_timeframe(df):
    """
    Система: Тренд (EMA) + Подтверждение (MACD) + Фильтр (RSI/объём) + Тайминг (SAR)
    """

    # --- EMA 20 / 50 / 100 / 200 ---
    df['EMA20']  = df['close'].ewm(span=20).mean()
    df['EMA50']  = df['close'].ewm(span=50).mean()
    df['EMA100'] = df['close'].ewm(span=100).mean()
    df['EMA200'] = df['close'].ewm(span=200).mean()

    # --- MA 50 / 200 ---
    df['MA50']  = df['close'].rolling(50).mean()
    df['MA200'] = df['close'].rolling(200).mean()

    # --- MACD (12/26/9) ---
    macd = ta.trend.MACD(df['close'], window_fast=12, window_slow=26, window_sign=9)
    df['MACD']       = macd.macd()
    df['MACD_signal'] = macd.macd_signal()
    df['MACD_hist']  = macd.macd_diff()

    # --- RSI 14 ---
    df['RSI'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()

    # --- ATR ---
    df['ATR'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()

    # --- Parabolic SAR (step=0.02, max=0.2) ---
    df['SAR'] = ta.trend.PSARIndicator(
        df['high'], df['low'], df['close'],
        step=0.02, max_step=0.2
    ).psar()

    # --- ADX ---
    df['ADX'] = ta.trend.ADXIndicator(df['high'], df['low'], df['close']).adx()

    last = df.iloc[-1]
    prev = df.iloc[-2]

    # ── ФЛЭТ ФИЛЬТР ──
    ema_diff_pct = abs(last['EMA20'] - last['EMA50']) / last['EMA50'] * 100
    if ema_diff_pct < 0.15:  # EMA20 ≈ EMA50 → флэт → молчим
        return "NO SIGNAL", last

    # ── ФИЛЬТР ПОСЛЕ СИЛЬНОГО ДВИЖЕНИЯ ──
    price_move = abs(last['close'] - prev['close'])
    if price_move > last['ATR'] * 1.5:
        return "NO SIGNAL", last

    # ══════════════════════════════════════
    # LONG CONDITIONS
    # Тренд: цена выше EMA200, EMA20 > EMA50
    # Подтверждение: MACD гистограмма растёт И выше 0
    # Фильтр: RSI 45–70, объём выше среднего
    # Тайминг: SAR ниже цены
    # ══════════════════════════════════════
    long_trend = (
        last['close'] > last['EMA200'] and
        last['close'] > last['MA200'] and
        last['EMA20'] > last['EMA50']
    )
    long_macd = (
        last['MACD_hist'] > 0 and
        last['MACD_hist'] > prev['MACD_hist']  # гистограмма растёт
    )
    long_rsi    = 45 < last['RSI'] < 70
    long_sar    = last['SAR'] < last['close']

    # ══════════════════════════════════════
    # SHORT CONDITIONS (зеркально)
    # ══════════════════════════════════════
    short_trend = (
        last['close'] < last['EMA200'] and
        last['close'] < last['MA200'] and
        last['EMA20'] < last['EMA50']
    )
    short_macd = (
        last['MACD_hist'] < 0 and
        last['MACD_hist'] < prev['MACD_hist']  # гистограмма падает
    )
    short_rsi   = 30 < last['RSI'] < 55
    short_sar   = last['SAR'] > last['close']

    if long_trend and long_macd and long_rsi and long_sar:
        return "LONG", last
    elif short_trend and short_macd and short_rsi and short_sar:
        return "SHORT", last
    else:
        return "NO SIGNAL", last


# =====================================================
# АНАЛИЗ МОНЕТЫ — СИСТЕМА 1H → 30m → 15m
# 1H = тренд (обязательно)
# 30m или 15m = подтверждение (хотя бы один)
# =====================================================
def analyze_symbol(symbol, timeframes):
    try:
        results = {}

        for tf in timeframes:
            limit = 250  # нужно для EMA200
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe=tf, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

            side, last = analyze_timeframe(df)
            results[tf] = {"side": side, "last": last, "df": df}

        # ── СТАРШИЙ ПОДТВЕРЖДАЕТ МЛАДШЕГО ──
        tf_1h  = timeframes[2]  # '1h'
        tf_30m = timeframes[1]  # '30m'
        tf_15m = timeframes[0]  # '15m'

        side_1h  = results[tf_1h]['side']
        side_30m = results[tf_30m]['side']
        side_15m = results[tf_15m]['side']

        # 1H обязателен
        if side_1h == "NO SIGNAL":
            return {"side": "NO SIGNAL"}

        # Хотя бы 30m или 15m подтверждает
        junior_confirms = (side_30m == side_1h) or (side_15m == side_1h)
        if not junior_confirms:
            return {"side": "NO SIGNAL"}

        final_side = side_1h

        # Берём данные с 1H для торгового плана
        last_1h = results[tf_1h]['last']
        df_1h   = results[tf_1h]['df']

        # --- Объём (с 1H) ---
        volume_data = analyze_volume(df_1h)
        if volume_data['volume_signal'] == "WEAK":
            return {"side": "NO SIGNAL"}

        # --- Паттерны свечей (с 15m — точка входа) ---
        df_15m = results[tf_15m]['df']
        patterns, pattern_signal = detect_candle_patterns(df_15m)

        if final_side == "LONG" and pattern_signal == "BEARISH":
            return {"side": "NO SIGNAL"}
        if final_side == "SHORT" and pattern_signal == "BULLISH":
            return {"side": "NO SIGNAL"}

        # --- Уровни ---
        supports, resistances = find_support_resistance(df_1h)
        liq_above, liq_below  = find_liquidity_levels(df_1h)

        # --- Торговый план ---
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
            "macd_hist":     last_1h['MACD_hist'],
            "tf_1h":         side_1h,
            "tf_30m":        side_30m,
            "tf_15m":        side_15m,
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
        return {"side": "NO SIGNAL"}


# =====================================================
# АСИНХРОННЫЙ ЗАПУСК ДЛЯ ВСЕХ МОНЕТ
# =====================================================
async def analyze_all_timeframes_async(symbol):
    loop = asyncio.get_running_loop()
    signal = await loop.run_in_executor(None, analyze_symbol, symbol, TIMEFRAMES)
    return signal


# =====================================================
# ФОРМАТИРОВАНИЕ СИГНАЛА
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

    patterns_text    = "\n   ".join(signal['patterns']) if signal['patterns'] else "—"
    supports_text    = " | ".join([f"{p:.4f}" for _, p in signal['supports']])    if signal['supports']    else "—"
    resistances_text = " | ".join([f"{p:.4f}" for _, p in signal['resistances']]) if signal['resistances'] else "—"
    liq_above_text   = " | ".join([f"{p:.4f}" for p in signal['liq_above']])      if signal['liq_above']   else "—"
    liq_below_text   = " | ".join([f"{p:.4f}" for p in signal['liq_below']])      if signal['liq_below']   else "—"

    return f"""
🚨 TRADE PLAN | {symbol_fmt} | {side}
{emoji} Подтверждение по таймфреймам:
   1H:  {tf_emoji(signal['tf_1h'])}  30M: {tf_emoji(signal['tf_30m'])}  15M: {tf_emoji(signal['tf_15m'])}

💰 Зона набора:
   {signal['entry_min']:.4f} — {signal['entry_max']:.4f}

🛑 Стоп-лосс:
   {signal['stop_loss']:.4f}

❌ Отмена идеи:
   H1 close {'<' if side == 'LONG' else '>'} {signal['invalidation']:.4f}

🎯 Тейки:
   TP1: {signal['tp1']:.4f}  (25% позиции)
   TP2: {signal['tp2']:.4f}  (50% позиции) — RR 1:2
   TP3: {signal['tp3']:.4f}  (25% позиции) — RR 1:3

📊 Индикаторы:
   RSI: {signal['rsi']:.1f}   ADX: {signal['adx']:.1f}
   EMA20: {signal['ema20']:.4f}
   EMA50: {signal['ema50']:.4f}
   EMA200: {signal['ema200']:.4f}
   MACD hist: {signal['macd_hist']:.6f}

📦 Объём: {vol['volume_emoji']} x{vol['volume_ratio']:.1f} от среднего

🕯 Паттерны:
   {patterns_text}

🏛 Поддержки:  {supports_text}
🏛 Сопротивления: {resistances_text}

💧 Ликвидность выше: {liq_above_text}
💧 Ликвидность ниже: {liq_below_text}

💵 Цена: {signal['current_price']:.4f}
"""