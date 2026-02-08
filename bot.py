import socket

# Принудительно используем IPv4
original_getaddrinfo = socket.getaddrinfo

def getaddrinfo_ipv4(*args, **kwargs):
    return [x for x in original_getaddrinfo(*args, **kwargs) if x[0].name == 'AF_INET']

socket.getaddrinfo = getaddrinfo_ipv4

import os
import asyncio
from aiogram import Bot, Dispatcher, types
from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.filters import Command
from dotenv import load_dotenv

load_dotenv()
# ----------------------
# Загружаем .env
# ----------------------
TOKEN = os.getenv("BOT_TOKEN")
OWNER_ID_ENV = os.getenv("OWNER_ID")
OWNER_ID = int(OWNER_ID_ENV) if OWNER_ID_ENV and OWNER_ID_ENV.strip() != "" else None

# Проверка токена
if not TOKEN:
    exit("Ошибка: BOT_TOKEN не найден в .env")

session = AiohttpSession(timeout=30)

# ----------------------
# Создаем бота и диспетчер
# ----------------------
bot = Bot( 
    token=TOKEN,
    session=session
    )


dp = Dispatcher()

# ----------------------
# Обработчик /start
# ----------------------


# ----------------------
# main для запуска polling
# ----------------------
async def main():
    print("Бот запущен и ждёт сообщений...")
    await dp.start_polling(bot)

# ----------------------
# Запуск
# ----------------------
if __name__ == "__main__":
    asyncio.run(main())



    import ccxt
import os
import pandas as pd
import ta
import asyncio
import time

from dotenv import load_dotenv
from aiogram import Bot, Dispatcher, types
from aiogram.utils import executor

# =====================================================
# ЗАГРУЗКА КЛЮЧЕЙ
# =====================================================

load_dotenv()

TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("OWNER_ID")
BYBIT_API_KEY = os.getenv("BYBIT_API_KEY")
BYBIT_API_SECRET = os.getenv("BYBIT_API_SECRET")

bot = Bot(token=TOKEN)
dp = Dispatcher(bot)

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

def signal_keyboard():
    """Создает клавиатуру c одной кнопкой"""
    keyboard = InlineKeyboardMarkup(row_width=1)
    keyboard.add(
        InlineKeyboardButton(
            text="📊 Получить сигнал",
            callback_data="get_best_signal"
        )
    )
    return keyboard
    
            

# cooldown защита от спама
signal_cooldown = {}
COOLDOWN_SECONDS = 1800  # 30 минут

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

    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=200)

    df = pd.DataFrame(
        ohlcv,
        columns=['timestamp','open','high','low','close','volume']
    )

    # индикаторы
    df['MA'] = df['close'].rolling(20).mean()
    df['EMA'] = df['close'].ewm(span=20).mean()

    bb = ta.volatility.BollingerBands(df['close'])
    df['bb_up'] = bb.bollinger_hband()
    df['bb_low'] = bb.bollinger_lband()

    df['SAR'] = ta.trend.PSARIndicator(
        df['high'], df['low'], df['close']
    ).psar()

    macd = ta.trend.MACD(df['close'])
    df['MACD'] = macd.macd()
    df['MACD_signal'] = macd.macd_signal()

    df['RSI'] = ta.momentum.RSIIndicator(df['close']).rsi()

    df['WR'] = ta.momentum.WilliamsRIndicator(
        df['high'], df['low'], df['close']
    ).williams_r()

    df['StochRSI'] = ta.momentum.StochRSIIndicator(
        df['close']
    ).stochrsi()

    df['ATR'] = ta.volatility.AverageTrueRange(
        df['high'], df['low'], df['close']
    ).average_true_range()

    df['ADX'] = ta.trend.ADXIndicator(
        df['high'], df['low'], df['close']
    ).adx()

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

    entry, sl, tp1, tp2, tp3 = build_trade_plan(
        last['close'],
        last['ATR'],
        side
    )

    return {
        "symbol": symbol,
        "side": side,
        "entry": entry,
        "sl": sl,
        "tp1": tp1,
        "tp2": tp2,
        "tp3": tp3
    }


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
# TELEGRAM КОМАНДЫ
# =====================================================
@dp.message_handler(commands=['start'])
async def send_start(message: types.Message):
    """Показывает кнопку при старте"""
    keyboard = signal_keyboard()
    await message.reply(
        "👋 Привет! Нажми кнопку, чтобы получить лучший торговый сигнал:",
        reply_markup=keyboard
    )

@dp.callback_query_handler(lambda c: c.data == 'get_best_signal')
async def send_best_signal(callback: types.CallbackQuery):
    """Отправляет лучший сигнал из всех криптовалют"""
    await callback.answer("Анализирую рынок...")
    
    now = time.time()
    
    best_signal = None
    best_score = -999
    
    # Проверяем все криптовалюты
    for symbol in symbols:
        if symbol in signal_cooldown:
            if now - signal_cooldown[symbol] < COOLDOWN_SECONDS:
                continue
        
        signal = await analyze_all_timeframes(symbol)
        
        if signal['side'] != "NO SIGNAL":
            # Считаем "силу" сигнала
            score = 0
            if signal['entry'] == '🟢': score += 3
            if signal['sl'] == '🟢': score += 2
            if signal['tp1'] == '🟢': score += 1
            if signal['tp2'] == '🟢': score += 1
            if signal['tp3'] == '🟢': score += 1
            
            if score > best_score:
                best_score = score
                best_signal = signal
                best_signal['symbol'] = symbol
    
    if best_signal:
        signal_cooldown[best_signal['symbol']] = now
        await callback.message.reply(best_signal['text'])
    else:
        await callback.message.reply("⏳ Сейчас нет сильных сигналов. Попробуйте позже.")

@dp.message_handler(commands=['signal'])
async def send_signal(message: types.Message):

    now = time.time()

    for symbol in symbols:

        if symbol in signal_cooldown:
            if now - signal_cooldown[symbol] < COOLDOWN_SECONDS:
                continue

        signal = analyze_all_timeframes(symbol)

        if signal['side'] != "NO SIGNAL":

            text = f"""
TRADE PLAN | {signal['symbol']} | {signal['side']}
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
# АВТОСКАН РЫНКА
# =====================================================

async def auto_scan():

    while True:

        now = time.time()

        for symbol in symbols:

            if symbol in signal_cooldown:
                if now - signal_cooldown[symbol] < COOLDOWN_SECONDS:
                    continue

            signal = analyze_all_timeframes(symbol)

            if signal['side'] != "NO SIGNAL":

                text = f"""
🚨 AUTO SIGNAL

{signal['symbol']} | {signal['side']}

Entry ≈ {signal['entry']:.4f}
SL → {signal['sl']:.4f}

TP1 → {signal['tp1']:.4f}
TP2 → {signal['tp2']:.4f}
TP3 → {signal['tp3']:.4f}
"""

                await bot.send_message(CHAT_ID, text)

                signal_cooldown[symbol] = now

        await asyncio.sleep(300)  # скан каждые 5 минут

# =====================================================

loop = asyncio.get_event_loop()
loop.create_task(auto_scan())

if __name__ == "__main__":
    executor.start_polling(dp, skip_updates=True)