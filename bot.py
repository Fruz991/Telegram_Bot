import socket
import os
import asyncio
import time
import ccxt
import pandas as pd
import ta

from aiogram import Bot, Dispatcher, types, F
from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.filters import Command
from dotenv import load_dotenv

# =====================================================
# IPv4
# =====================================================
original_getaddrinfo = socket.getaddrinfo

def getaddrinfo_ipv4(*args, **kwargs):
    return [x for x in original_getaddrinfo(*args, **kwargs) if x[0].name == 'AF_INET']

socket.getaddrinfo = getaddrinfo_ipv4

# =====================================================
# ЗАГРУЗКА КЛЮЧЕЙ
# =====================================================
load_dotenv()

TOKEN = os.getenv("BOT_TOKEN")
OWNER_ID_ENV = os.getenv("OWNER_ID")
OWNER_ID = int(OWNER_ID_ENV) if OWNER_ID_ENV and OWNER_ID_ENV.strip() != "" else None
BYBIT_API_KEY = os.getenv("BYBIT_API_KEY")
BYBIT_API_SECRET = os.getenv("BYBIT_API_SECRET")

if not TOKEN:
    exit("Ошибка: BOT_TOKEN не найден в .env")

# =====================================================
# БОТ
# =====================================================
session = AiohttpSession(timeout=30)
bot = Bot(token=TOKEN, session=session)
dp = Dispatcher()

# =====================================================
# БИРЖА
# =====================================================
exchange = ccxt.bybit({
    "apiKey": BYBIT_API_KEY,
    "secret": BYBIT_API_SECRET,
    "enableRateLimit": True
})

# =====================================================
# НАСТРОЙКИ
# =====================================================
symbols = [
    "BTC/USDT","SOL/USDT","ETH/USDT","SUI/USDT","LTC/USDT",
    "BNB/USDT","WIF/USDT","ADA/USDT","ATOM/USDT","ZEC/USDT",
    "ENA/USDT","NEAR/USDT","OP/USDT"
]

timeframes = ['15m','30m','1h']
signal_cooldown = {}
COOLDOWN_SECONDS = 1800  # 30 минут

# =====================================================
# КЛАВИАТУРА
# =====================================================
def signal_keyboard():
    keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="📊 Получить сигнал", callback_data="get_best_signal")]
    ])
    return keyboard

# =====================================================
# RR + ATR
# =====================================================
def build_trade_plan(price, atr, side):
    risk = atr * 1.2

    if side == "LONG":
        entry = price
        sl = price - risk
        tp1 = price + risk * 2
        tp2 = price + risk * 2.5
        tp3 = price + risk * 3
    else:
        entry = price
        sl = price + risk
        tp1 = price - risk * 2
        tp2 = price - risk * 2.5
        tp3 = price - risk * 3

    return entry, sl, tp1, tp2, tp3

# =====================================================
# АНАЛИЗ МОНЕТЫ
# =====================================================
def analyze_symbol(symbol, timeframe):
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=200)
        df = pd.DataFrame(ohlcv, columns=['timestamp','open','high','low','close','volume'])

        # индикаторы
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
        df['WR'] = ta.momentum.WilliamsRIndicator(df['high'], df['low'], df['close']).williams_r()
        df['StochRSI'] = ta.momentum.StochRSIIndicator(df['close']).stochrsi()
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

        entry, sl, tp1, tp2, tp3 = build_trade_plan(last['close'], last['ATR'], side)

        return {
            "symbol": symbol,
            "side": side,
            "entry": entry,
            "sl": sl,
            "tp1": tp1,
            "tp2": tp2,
            "tp3": tp3
        }
    except Exception as e:
        print(f"Ошибка анализа {symbol}: {e}")
        return {"side": "NO SIGNAL"}

# =====================================================
# ПРОВЕРКА ВСЕХ ТФ
# =====================================================
def analyze_all_timeframes(symbol):
    signals = [analyze_symbol(symbol, tf) for tf in timeframes]
    sides = [s['side'] for s in signals if 'side' in s]

    if len(sides) == 3 and all(s == "LONG" for s in sides):
        return signals[0]
    if len(sides) == 3 and all(s == "SHORT" for s in sides):
        return signals[0]

    return {"side": "NO SIGNAL"}

# =====================================================
# КОМАНДЫ
# =====================================================
@dp.message(Command("start"))
async def send_start(message: types.Message):
    keyboard = signal_keyboard()
    await message.reply(
        "👋 Привет! Нажми кнопку, чтобы получить лучший торговый сигнал:",
        reply_markup=keyboard
    )

@dp.callback_query(F.data == "get_best_signal")
async def send_best_signal(callback: types.CallbackQuery):
    await callback.answer("Анализирую рынок...")
    
    now = time.time()
    best_signal = None
    
    for symbol in symbols:
        if symbol in signal_cooldown:
            if now - signal_cooldown[symbol] < COOLDOWN_SECONDS:
                continue
        
        signal = analyze_all_timeframes(symbol)
        
        if signal['side'] != "NO SIGNAL":
            best_signal = signal
            best_signal['symbol'] = symbol
            break
    
    if best_signal:
        text = f"""
🚨 TRADE PLAN | {best_signal['symbol']} | {best_signal['side']}
TF: 15M / 30M / 1H

Entry ≈ {best_signal['entry']:.4f}
SL → {best_signal['sl']:.4f}

TP1 → {best_signal['tp1']:.4f}
TP2 → {best_signal['tp2']:.4f}
TP3 → {best_signal['tp3']:.4f}
"""
        signal_cooldown[best_signal['symbol']] = now
        await callback.message.answer(text)
    else:
        await callback.message.answer("⏳ Сейчас нет сильных сигналов. Попробуйте позже.")

@dp.message(Command("signal"))
async def send_signal(message: types.Message):
    now = time.time()

    for symbol in symbols:
        if symbol in signal_cooldown:
            if now - signal_cooldown[symbol] < COOLDOWN_SECONDS:
                continue

        signal = analyze_all_timeframes(symbol)

        if signal['side'] != "NO SIGNAL":
            text = f"""
🚨 TRADE PLAN | {signal['symbol']} | {signal['side']}
TF: 15M / 30M / 1H

Entry ≈ {signal['entry']:.4f}
SL → {signal['sl']:.4f}

TP1 → {signal['tp1']:.4f}
TP2 → {signal['tp2']:.4f}
TP3 → {signal['tp3']:.4f}
"""
            signal_cooldown[symbol] = now
            await message.reply(text)
            return

    await message.reply("Сигналов сейчас нет.")

# =====================================================
# АВТОСКАН
# =====================================================
async def auto_scan():
    while True:
        now = time.time()
        for symbol in symbols:
            if symbol in signal_cooldown:
                if now - signal_cooldown[symbol] < COOLDOWN_SECONDS:
                    continue

            signal = analyze_all_timeframes(symbol)

            if signal['side'] != "NO SIGNAL" and OWNER_ID:
                text = f"""
🚨 AUTO SIGNAL

{signal['symbol']} | {signal['side']}

Entry ≈ {signal['entry']:.4f}
SL → {signal['sl']:.4f}

TP1 → {signal['tp1']:.4f}
TP2 → {signal['tp2']:.4f}
TP3 → {signal['tp3']:.4f}
"""
                await bot.send_message(OWNER_ID, text)
                signal_cooldown[symbol] = now

        await asyncio.sleep(300)

# =====================================================
# ЗАПУСК
# =====================================================
async def main():
    print("Бот запущен и ждёт сообщений...")
    asyncio.create_task(auto_scan())
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())