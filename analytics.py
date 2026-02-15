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
    else:  # SHORT
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
        "risk_reward_ratio": "1:2 - 1:3"
    }


# =====================================================
# УРОВНИ ПОДДЕРЖКИ И СОПРОТИВЛЕНИЯ
# =====================================================
def find_support_resistance(df, window=10, num_levels=3):
    """Находит ключевые уровни поддержки и сопротивления"""
    highs = df['high'].values
    lows = df['low'].values
    levels = []

    for i in range(window, len(df) - window):
        # Локальный максимум (сопротивление)
        if highs[i] == max(highs[i - window:i + window]):
            levels.append(("resistance", highs[i]))
        # Локальный минимум (поддержка)
        if lows[i] == min(lows[i - window:i + window]):
            levels.append(("support", lows[i]))

    # Кластеризуем близкие уровни
    clustered = []
    used = set()
    price_range = (df['high'].max() - df['low'].min()) * 0.01  # 1% диапазон

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

    # Сортируем по силе уровня (количество касаний)
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


def nearest_level_distance(price, supports, resistances):
    """Возвращает расстояние до ближайшего уровня в %"""
    all_levels = [p for _, p in supports + resistances]
    if not all_levels:
        return None
    nearest = min(all_levels, key=lambda x: abs(x - price))
    distance_pct = abs(nearest - price) / price * 100
    return distance_pct, nearest


# =====================================================
# ПАТТЕРНЫ СВЕЧЕЙ
# =====================================================
def detect_candle_patterns(df):
    """Распознаёт паттерны свечей на последних свечах"""
    patterns = []
    last = df.iloc[-1]
    prev = df.iloc[-2]

    body = abs(last['close'] - last['open'])
    upper_shadow = last['high'] - max(last['close'], last['open'])
    lower_shadow = min(last['close'], last['open']) - last['low']
    total_range = last['high'] - last['low']

    if total_range == 0:
        return patterns, "NEUTRAL"

    # --- МОЛОТ (бычий) ---
    if (lower_shadow >= body * 2 and
        upper_shadow <= body * 0.3 and
        last['close'] > last['open']):
        patterns.append("🔨 Молот (бычий)")

    # --- ПЕРЕВЁРНУТЫЙ МОЛОТ (бычий) ---
    if (upper_shadow >= body * 2 and
        lower_shadow <= body * 0.3 and
        last['close'] > last['open']):
        patterns.append("🔨 Перевёрнутый молот (бычий)")

    # --- ПАДАЮЩАЯ ЗВЕЗДА (медвежий) ---
    if (upper_shadow >= body * 2 and
        lower_shadow <= body * 0.3 and
        last['close'] < last['open']):
        patterns.append("⭐ Падающая звезда (медвежий)")

    # --- ДОДЖИ ---
    if body <= total_range * 0.1:
        patterns.append("➖ Доджи (неопределённость)")

    # --- БЫЧЬЕ ПОГЛОЩЕНИЕ ---
    if (last['close'] > last['open'] and
        prev['close'] < prev['open'] and
        last['open'] < prev['close'] and
        last['close'] > prev['open']):
        patterns.append("📈 Бычье поглощение")

    # --- МЕДВЕЖЬЕ ПОГЛОЩЕНИЕ ---
    if (last['close'] < last['open'] and
        prev['close'] > prev['open'] and
        last['open'] > prev['close'] and
        last['close'] < prev['open']):
        patterns.append("📉 Медвежье поглощение")

    # --- ПИНБАР БЫЧИЙ ---
    if (lower_shadow >= total_range * 0.6 and
        body <= total_range * 0.3):
        patterns.append("📌 Пинбар (бычий)")

    # --- ПИНБАР МЕДВЕЖИЙ ---
    if (upper_shadow >= total_range * 0.6 and
        body <= total_range * 0.3):
        patterns.append("📌 Пинбар (медвежий)")

    # Определяем общий сигнал паттернов
    bullish_words = ["бычий", "Молот", "Пинбар (бычий)", "Поглощение (бычий)"]
    bearish_words = ["медвежий", "Звезда", "Поглощение (медвежий)", "Пинбар (медвежий)"]

    bullish_count = sum(1 for p in patterns if any(w in p for w in bullish_words))
    bearish_count = sum(1 for p in patterns if any(w in p for w in bearish_words))

    if bullish_count > bearish_count:
        pattern_signal = "BULLISH"
    elif bearish_count > bullish_count:
        pattern_signal = "BEARISH"
    else:
        pattern_signal = "NEUTRAL"

    return patterns, pattern_signal


# =====================================================
# АНАЛИЗ ОБЪЁМА И ЛИКВИДНОСТИ
# =====================================================
def analyze_volume(df):
    """Анализирует объём свечей"""
    avg_volume = df['volume'].rolling(20).mean().iloc[-1]
    last_volume = df['volume'].iloc[-1]
    volume_ratio = last_volume / avg_volume if avg_volume > 0 else 1

    if volume_ratio >= 2.0:
        volume_signal = "STRONG"    # Очень сильный объём
        volume_emoji = "🔥"
    elif volume_ratio >= 1.3:
        volume_signal = "ABOVE"     # Выше среднего
        volume_emoji = "📊"
    elif volume_ratio >= 0.7:
        volume_signal = "NORMAL"    # Нормальный
        volume_emoji = "➡️"
    else:
        volume_signal = "WEAK"      # Слабый объём
        volume_emoji = "⚠️"

    return {
        "volume_ratio": volume_ratio,
        "volume_signal": volume_signal,
        "volume_emoji": volume_emoji,
        "avg_volume": avg_volume,
        "last_volume": last_volume
    }


def find_liquidity_levels(df, lookback=50):
    """Находит уровни ликвидности (зоны скопления стопов)"""
    recent = df.tail(lookback)

    # Зоны ликвидности — это swing high/low где скапливаются стопы
    swing_highs = []
    swing_lows = []

    for i in range(2, len(recent) - 2):
        h = recent['high'].iloc[i]
        l = recent['low'].iloc[i]

        # Swing high
        if (h > recent['high'].iloc[i-1] and h > recent['high'].iloc[i-2] and
            h > recent['high'].iloc[i+1] and h > recent['high'].iloc[i+2]):
            swing_highs.append(h)

        # Swing low
        if (l < recent['low'].iloc[i-1] and l < recent['low'].iloc[i-2] and
            l < recent['low'].iloc[i+1] and l < recent['low'].iloc[i+2]):
            swing_lows.append(l)

    current_price = recent['close'].iloc[-1]

    # Ближайшие уровни ликвидности
    liq_above = sorted([h for h in swing_highs if h > current_price])[:2]
    liq_below = sorted([l for l in swing_lows if l < current_price], reverse=True)[:2]

    return liq_above, liq_below


# =====================================================
# АНАЛИЗ ОДНОЙ МОНЕТЫ НА ОДНОМ ТФ
# =====================================================
def analyze_symbol(symbol, timeframe):
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=150)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

        # --- Базовые индикаторы ---
        df['MA'] = df['close'].rolling(20).mean()
        df['EMA'] = df['close'].ewm(span=20).mean()

        bb = ta.volatility.BollingerBands(df['close'])
        df['bb_up'] = bb.bollinger_hband()
        df['bb_low'] = bb.bollinger_lband()

        df['SAR'] = ta.trend.PSARIndicator(df['high'], df['low'], df['close']).psar()

        macd = ta.trend.MACD(df['close'])
        df['MACD'] = macd.macd()
        df['MACD_signal'] = macd.macd_signal()

        df['RSI'] = ta.momentum.RSIIndicator(df['close']).rsi()
        df['ATR'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
        df['ADX'] = ta.trend.ADXIndicator(df['high'], df['low'], df['close']).adx()

        # --- Stochastic RSI ---
        stoch_rsi = ta.momentum.StochRSIIndicator(df['close'])
        df['stoch_rsi_k'] = stoch_rsi.stochrsi_k()
        df['stoch_rsi_d'] = stoch_rsi.stochrsi_d()

        # --- OBV ---
        df['OBV'] = ta.volume.OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()
        df['OBV_MA'] = df['OBV'].rolling(20).mean()

        # --- Ichimoku ---
        ichimoku = ta.trend.IchimokuIndicator(df['high'], df['low'])
        df['ichi_a'] = ichimoku.ichimoku_a()
        df['ichi_b'] = ichimoku.ichimoku_b()
        df['ichi_base'] = ichimoku.ichimoku_base_line()
        df['ichi_conv'] = ichimoku.ichimoku_conversion_line()

        last = df.iloc[-1]
        trend_strong = last['ADX'] > 20

        # --- OBV тренд ---
        obv_bullish = last['OBV'] > last['OBV_MA']
        obv_bearish = last['OBV'] < last['OBV_MA']

        # --- Stoch RSI сигнал ---
        stoch_oversold = last['stoch_rsi_k'] < 0.2 and last['stoch_rsi_d'] < 0.2
        stoch_overbought = last['stoch_rsi_k'] > 0.8 and last['stoch_rsi_d'] > 0.8

        # --- Ichimoku сигнал ---
        ichi_bullish = (last['close'] > last['ichi_a'] and
                        last['close'] > last['ichi_b'] and
                        last['ichi_conv'] > last['ichi_base'])
        ichi_bearish = (last['close'] < last['ichi_a'] and
                        last['close'] < last['ichi_b'] and
                        last['ichi_conv'] < last['ichi_base'])

        # --- Базовые условия ---
        long_cond = (
            trend_strong and
            last['close'] > last['MA'] and
            last['close'] > last['EMA'] and
            last['SAR'] < last['close'] and
            last['MACD'] > last['MACD_signal'] and
            last['RSI'] < 45 and
            obv_bullish and
            stoch_oversold and
            ichi_bullish
        )

        short_cond = (
            trend_strong and
            last['close'] < last['MA'] and
            last['close'] < last['EMA'] and
            last['SAR'] > last['close'] and
            last['MACD'] < last['MACD_signal'] and
            last['RSI'] > 55 and
            obv_bearish and
            stoch_overbought and
            ichi_bearish
        )

        if long_cond:
            side = "LONG"
        elif short_cond:
            side = "SHORT"
        else:
            return {"side": "NO SIGNAL"}

        # --- Паттерны свечей ---
        patterns, pattern_signal = detect_candle_patterns(df)

        # Паттерн должен подтверждать направление
        if side == "LONG" and pattern_signal == "BEARISH":
            return {"side": "NO SIGNAL"}
        if side == "SHORT" and pattern_signal == "BULLISH":
            return {"side": "NO SIGNAL"}

        # --- Объём ---
        volume_data = analyze_volume(df)

        # Слабый объём — пропускаем сигнал
        if volume_data["volume_signal"] == "WEAK":
            return {"side": "NO SIGNAL"}

        # --- Уровни поддержки/сопротивления ---
        supports, resistances = find_support_resistance(df)
        liq_above, liq_below = find_liquidity_levels(df)

        trade_plan = build_advanced_trade_plan(last['close'], last['ATR'], side)

        return {
            "symbol": symbol,
            "side": side,
            "current_price": last['close'],
            "rsi": last['RSI'],
            "adx": last['ADX'],
            "stoch_k": last['stoch_rsi_k'],
            "stoch_d": last['stoch_rsi_d'],
            "obv_trend": "↑ Бычий" if obv_bullish else "↓ Медвежий",
            "ichi_signal": "✅ Выше облака" if ichi_bullish else "❌ Ниже облака",
            "patterns": patterns,
            "pattern_signal": pattern_signal,
            "volume_data": volume_data,
            "supports": supports,
            "resistances": resistances,
            "liq_above": liq_above,
            "liq_below": liq_below,
            **trade_plan
        }
    except Exception as e:
        print(f"Ошибка анализа {symbol}: {e}")
        return {"side": "NO SIGNAL"}


# =====================================================
# АНАЛИЗ ВСЕХ ТАЙМФРЕЙМОВ ПАРАЛЛЕЛЬНО
# =====================================================
async def analyze_all_timeframes_async(symbol):
    loop = asyncio.get_running_loop()
    tasks = [loop.run_in_executor(None, analyze_symbol, symbol, tf) for tf in TIMEFRAMES]
    signals = await asyncio.gather(*tasks)
    sides = [s['side'] for s in signals if 'side' in s]

    if len(sides) == 3 and all(s == "LONG" for s in sides):
        return signals[0]
    if len(sides) == 3 and all(s == "SHORT" for s in sides):
        return signals[0]

    return {"side": "NO SIGNAL"}


# =====================================================
# ФОРМАТИРОВАНИЕ СИГНАЛА
# =====================================================
def format_signal(signal):
    symbol_formatted = signal['symbol'].replace('/', '')
    side = signal['side']
    emoji = "📈" if side == "LONG" else "📉"

    # Паттерны
    patterns_text = "\n   ".join(signal['patterns']) if signal['patterns'] else "Паттернов нет"

    # Уровни поддержки
    supports_text = "\n   ".join([f"{p:.4f}" for _, p in signal['supports']]) if signal['supports'] else "—"
    resistances_text = "\n   ".join([f"{p:.4f}" for _, p in signal['resistances']]) if signal['resistances'] else "—"

    # Ликвидность
    liq_above_text = " | ".join([f"{p:.4f}" for p in signal['liq_above']]) if signal['liq_above'] else "—"
    liq_below_text = " | ".join([f"{p:.4f}" for p in signal['liq_below']]) if signal['liq_below'] else "—"

    # Объём
    vol = signal['volume_data']
    volume_text = f"{vol['volume_emoji']} x{vol['volume_ratio']:.1f} от среднего"

    return f"""
🚨 TRADE PLAN | {symbol_formatted} | {side}
TF: 15M/30M/1H | Сигнал: {emoji}

💰 Зона набора:
   {signal['entry_min']:.4f} - {signal['entry_max']:.4f}

🛑 Стоп-лосс:
   {signal['stop_loss']:.4f}

❌ Отмена идеи:
   H1 close {'<' if side == 'LONG' else '>'} {signal['invalidation']:.4f}

🎯 Зона фиксации:
   TP1: {signal['tp1']:.4f} (25% позиции)
   TP2: {signal['tp2']:.4f} (50% позиции) - RR 1:2
   TP3: {signal['tp3']:.4f} (25% позиции) - RR 1:3

📊 Индикаторы:
   RSI: {signal['rsi']:.1f}
   ADX: {signal['adx']:.1f}
   Stoch RSI: K={signal['stoch_k']:.2f} D={signal['stoch_d']:.2f}
   OBV: {signal['obv_trend']}
   Ichimoku: {signal['ichi_signal']}

🕯 Паттерны свечей:
   {patterns_text}

📦 Объём:
   {volume_text}

🏛 Поддержки:
   {supports_text}

🏛 Сопротивления:
   {resistances_text}

💧 Ликвидность выше:
   {liq_above_text}
💧 Ликвидность ниже:
   {liq_below_text}

💵 Текущая цена: {signal['current_price']:.4f}
"""