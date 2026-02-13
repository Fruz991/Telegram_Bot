import asyncio
import os
import socket
import ccxt
import pandas as pd
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
# АНАЛИЗ ОДНОЙ МОНЕТЫ НА ОДНОМ ТФ
# =====================================================
def analyze_symbol(symbol, timeframe):
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=100)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

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

        last = df.iloc[-1]
        trend_strong = last['ADX'] > 20

        long_cond = (
            trend_strong and
            last['close'] > last['MA'] and
            last['close'] > last['EMA'] and
            last['SAR'] < last['close'] and
            last['MACD'] > last['MACD_signal'] and
            last['RSI'] < 40
        )

        short_cond = (
            trend_strong and
            last['close'] < last['MA'] and
            last['close'] < last['EMA'] and
            last['SAR'] > last['close'] and
            last['MACD'] < last['MACD_signal'] and
            last['RSI'] > 60
        )

        if long_cond:
            side = "LONG"
        elif short_cond:
            side = "SHORT"
        else:
            return {"side": "NO SIGNAL"}

        trade_plan = build_advanced_trade_plan(last['close'], last['ATR'], side)

        return {
            "symbol": symbol,
            "side": side,
            "current_price": last['close'],
            "rsi": last['RSI'],
            "adx": last['ADX'],
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

📊 Дополнительные данные:
   RSI: {signal['rsi']:.1f}
   ADX: {signal['adx']:.1f}
   Текущая цена: {signal['current_price']:.4f}
"""


# =====================================================
# ФОРМАТИРОВАНИЕ ОТЧЕТА О СДЕЛКЕ
# =====================================================
def format_trade_report(symbol, result_r, tp1, tp2, tp3, exit_reason, comment=""):
    tp1_status = "✔" if tp1 else "❌"
    tp2_status = "✔" if tp2 else "❌"
    tp3_status = "✔" if tp3 else "❌"
    result_emoji = "📈" if result_r > 0 else "📉"

    text = f"""
📊 Итог сделки {symbol}
Результат: {result_emoji} {result_r:+.1f}R
TP1 {tp1_status} TP2 {tp2_status} TP3 {tp3_status}
Причина выхода: {exit_reason}
"""
    if comment:
        text += f"Комментарий трейдера: {comment}"

    return text