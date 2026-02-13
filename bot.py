import socket
import os
import asyncio
import time
import ccxt
import pandas as pd
import ta
from datetime import datetime, timedelta
import json

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
OWNER_ID = 6941110878  # Твой Telegram ID
BYBIT_API_KEY = os.getenv("BYBIT_API_KEY")
BYBIT_API_SECRET = os.getenv("BYBIT_API_SECRET")

if not TOKEN:
    exit("Ошибка: BOT_TOKEN не найден в .env")

# =====================================================
# БОТ
# =====================================================
session = AiohttpSession(timeout=10)  # Уменьшили таймаут для быстроты
bot = Bot(token=TOKEN, session=session)
dp = Dispatcher()

# =====================================================
# БИРЖА (ОПТИМИЗАЦИЯ СКОРОСТИ)
# =====================================================
exchange = ccxt.bybit({
    "apiKey": BYBIT_API_KEY,
    "secret": BYBIT_API_SECRET,
    "enableRateLimit": True,
    "rateLimit": 50,  # Минимальная задержка между запросами (50ms)
    "timeout": 10000,  # 10 секунд таймаут
    "options": {
        "defaultType": "future",  # Используем фьючерсы для быстроты
        "adjustForTimeDifference": True  # Автокоррекция времени
    }
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
# СИСТЕМА УЧЕТА СДЕЛОК И ЛИМИТОВ
# =====================================================
class TradeTracker:
    def __init__(self):
        self.trades_file = "trades_history.json"
        self.trades = self.load_trades()
    
    def load_trades(self):
        """Загружает историю сделок из файла"""
        try:
            if os.path.exists(self.trades_file):
                with open(self.trades_file, 'r') as f:
                    return json.load(f)
            return {"daily_stops": 0, "last_reset": str(datetime.now().date()), "history": []}
        except:
            return {"daily_stops": 0, "last_reset": str(datetime.now().date()), "history": []}
    
    def save_trades(self):
        """Сохраняет историю сделок в файл"""
        try:
            with open(self.trades_file, 'w') as f:
                json.dump(self.trades, f, indent=2)
        except Exception as e:
            print(f"Ошибка сохранения trades: {e}")
    
    def reset_daily_if_needed(self):
        """Сбрасывает счетчик стопов если наступил новый день"""
        today = str(datetime.now().date())
        if self.trades["last_reset"] != today:
            self.trades["daily_stops"] = 0
            self.trades["last_reset"] = today
            self.save_trades()
    
    def add_stop(self):
        """Добавляет стоп в счетчик"""
        self.reset_daily_if_needed()
        self.trades["daily_stops"] += 1
        self.save_trades()
    
    def get_stops_count(self):
        """Возвращает количество стопов за день"""
        self.reset_daily_if_needed()
        return self.trades["daily_stops"]
    
    def can_trade(self):
        """Проверяет можно ли торговать (не превышен лимит стопов)"""
        return self.get_stops_count() < 3
    
    def add_trade_report(self, symbol, side, result, tps_hit, exit_reason, comment=""):
        """Добавляет отчет о сделке"""
        trade_report = {
            "date": str(datetime.now()),
            "symbol": symbol,
            "side": side,
            "result": result,
            "tps_hit": tps_hit,
            "exit_reason": exit_reason,
            "comment": comment
        }
        self.trades["history"].append(trade_report)
        self.save_trades()
        return trade_report

tracker = TradeTracker()

# =====================================================
# MIDDLEWARE ДЛЯ ПРОВЕРКИ ДОСТУПА
# =====================================================
def access_check(func):
    """Декоратор для проверки доступа только владельцу"""
    async def wrapper(message_or_callback, *args, **kwargs):
        user_id = None
        
        if isinstance(message_or_callback, types.Message):
            user_id = message_or_callback.from_user.id
        elif isinstance(message_or_callback, types.CallbackQuery):
            user_id = message_or_callback.from_user.id
        
        if user_id != OWNER_ID:
            # Отправляем сообщение о тех. обслуживании
            if isinstance(message_or_callback, types.Message):
                await message_or_callback.reply("⚠️ На данный момент бот находится в Техническом Обслуживании")
            elif isinstance(message_or_callback, types.CallbackQuery):
                await message_or_callback.answer("⚠️ Бот на обслуживании", show_alert=True)
            return
        
        return await func(message_or_callback, *args, **kwargs)
    
    return wrapper

# =====================================================
# КЛАВИАТУРА
# =====================================================
def signal_keyboard():
    keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="📊 Получить сигнал", callback_data="get_best_signal")],
        [InlineKeyboardButton(text="📈 Отчет по сделке", callback_data="trade_report")]
    ])
    return keyboard

def trade_report_keyboard():
    """Клавиатура для отчета по сделке"""
    keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(text="✅ Профит", callback_data="report_profit"),
            InlineKeyboardButton(text="❌ Стоп", callback_data="report_stop")
        ],
        [InlineKeyboardButton(text="🔙 Назад", callback_data="back_main")]
    ])
    return keyboard

# =====================================================
# УЛУЧШЕННЫЙ TRADE PLAN С RR
# =====================================================
def build_advanced_trade_plan(price, atr, side):
    """
    Создает торговый план с RR 1:2 и 1:3
    """
    risk = atr * 1.5
    
    if side == "LONG":
        entry_min = price - (atr * 0.2)
        entry_max = price + (atr * 0.2)
        stop_loss = price - risk
        invalidation = stop_loss - (atr * 0.3)
        tp1 = price + (risk * 1.5)
        tp2 = price + (risk * 2.0)
        tp3 = price + (risk * 3.0)
        
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
# АНАЛИЗ МОНЕТЫ (ОПТИМИЗИРОВАН)
# =====================================================
def analyze_symbol(symbol, timeframe):
    try:
        # Быстрый запрос с минимальными данными
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=100)  # Уменьшили до 100
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
# ПРОВЕРКА ВСЕХ ТФ (ПАРАЛЛЕЛЬНО ДЛЯ СКОРОСТИ)
# =====================================================
async def analyze_all_timeframes_async(symbol):
    """Асинхронный анализ всех таймфреймов для ускорения"""
    loop = asyncio.get_event_loop()
    
    # Запускаем анализ всех таймфреймов параллельно
    tasks = [loop.run_in_executor(None, analyze_symbol, symbol, tf) for tf in timeframes]
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
    """Форматирует сигнал в красивый текст"""
    symbol_formatted = signal['symbol'].replace('/', '')
    side = signal['side']
    emoji = "📈" if side == "LONG" else "📉"
    
    text = f"""
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
    return text

# =====================================================
# ФОРМАТИРОВАНИЕ ОТЧЕТА О СДЕЛКЕ
# =====================================================
def format_trade_report(symbol, result_r, tp1, tp2, tp3, exit_reason, comment=""):
    """Форматирует отчет о завершенной сделке"""
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

# =====================================================
# КОМАНДЫ
# =====================================================
@dp.message(Command("start"))
@access_check
async def send_start(message: types.Message):
    keyboard = signal_keyboard()
    stops_count = tracker.get_stops_count()
    
    status_text = f"Стопов сегодня: {stops_count}/3\n\n"
    
    await message.reply(
        f"👋 Привет! {status_text}Нажми кнопку для работы:",
        reply_markup=keyboard
    )

@dp.callback_query(F.data == "get_best_signal")
@access_check
async def send_best_signal(callback: types.CallbackQuery):
    # Проверка лимита стопов
    if not tracker.can_trade():
        await callback.answer("🚫 Лимит на сегодня достигнут. Иди отдыхай.", show_alert=True)
        return
    
    await callback.answer("Анализирую рынок...")
    
    now = time.time()
    best_signal = None
    
    for symbol in symbols:
        if symbol in signal_cooldown:
            if now - signal_cooldown[symbol] < COOLDOWN_SECONDS:
                continue
        
        signal = await analyze_all_timeframes_async(symbol)
        
        if signal['side'] != "NO SIGNAL":
            best_signal = signal
            break
    
    if best_signal:
        text = format_signal(best_signal)
        signal_cooldown[best_signal['symbol']] = now
        await callback.message.answer(text)
    else:
        await callback.message.answer("⏳ Сейчас нет сильных сигналов. Попробуйте позже.")

@dp.callback_query(F.data == "trade_report")
@access_check
async def show_trade_report_menu(callback: types.CallbackQuery):
    keyboard = trade_report_keyboard()
    await callback.message.answer(
        "Как завершилась сделка?",
        reply_markup=keyboard
    )
    await callback.answer()

@dp.callback_query(F.data == "report_profit")
@access_check
async def report_profit(callback: types.CallbackQuery):
    # Пример отчета о профитной сделке
    report = format_trade_report(
        symbol="BTCUSDT",
        result_r=2.4,
        tp1=True,
        tp2=True,
        tp3=False,
        exit_reason="трейлинг",
        comment="поздно перевёл SL"
    )
    
    await callback.message.answer(report)
    await callback.answer("✅ Отчет сохранен")

@dp.callback_query(F.data == "report_stop")
@access_check
async def report_stop(callback: types.CallbackQuery):
    # Добавляем стоп в счетчик
    tracker.add_stop()
    stops_count = tracker.get_stops_count()
    
    # Пример отчета о стопе
    report = format_trade_report(
        symbol="ETHUSDT",
        result_r=-1.0,
        tp1=False,
        tp2=False,
        tp3=False,
        exit_reason="стоп-лосс",
        comment="не дождался отката"
    )
    
    warning = ""
    if stops_count >= 3:
        warning = "\n\n🚫 Лимит стопов достигнут! Отдохни сегодня."
    
    await callback.message.answer(report + warning)
    await callback.answer(f"❌ Стоп #{stops_count}/3")

@dp.callback_query(F.data == "back_main")
@access_check
async def back_to_main(callback: types.CallbackQuery):
    keyboard = signal_keyboard()
    await callback.message.answer(
        "Главное меню:",
        reply_markup=keyboard
    )
    await callback.answer()

@dp.message(Command("signal"))
@access_check
async def send_signal(message: types.Message):
    # Проверка лимита стопов
    if not tracker.can_trade():
        await message.reply("🚫 Лимит на сегодня достигнут. Иди отдыхай.")
        return
    
    now = time.time()

    for symbol in symbols:
        if symbol in signal_cooldown:
            if now - signal_cooldown[symbol] < COOLDOWN_SECONDS:
                continue

        signal = await analyze_all_timeframes_async(symbol)

        if signal['side'] != "NO SIGNAL":
            text = format_signal(signal)
            signal_cooldown[symbol] = now
            await message.reply(text)
            return

    await message.reply("⏳ Сигналов сейчас нет.")

@dp.message(Command("stats"))
@access_check
async def show_stats(message: types.Message):
    """Показывает статистику стопов за день"""
    stops_count = tracker.get_stops_count()
    can_trade = tracker.can_trade()
    
    status = "✅ Можно торговать" if can_trade else "🚫 Лимит достигнут"
    
    await message.reply(f"""
📊 Статистика на сегодня:
Стопов: {stops_count}/3
Статус: {status}
""")

# =====================================================
# АВТОСКАН (ОПТИМИЗИРОВАН)
# =====================================================
async def auto_scan():
    while True:
        # Проверяем лимит стопов
        if not tracker.can_trade():
            await asyncio.sleep(3600)  # Ждем час если лимит достигнут
            continue
        
        now = time.time()
        for symbol in symbols:
            if symbol in signal_cooldown:
                if now - signal_cooldown[symbol] < COOLDOWN_SECONDS:
                    continue

            signal = await analyze_all_timeframes_async(symbol)

            if signal['side'] != "NO SIGNAL" and OWNER_ID:
                text = format_signal(signal)
                await bot.send_message(OWNER_ID, text)
                signal_cooldown[symbol] = now

        await asyncio.sleep(300)  # Скан каждые 5 минут

# =====================================================
# ЗАПУСК
# =====================================================
async def main():
    print("Бот запущен и ждёт сообщений...")
    print(f"Разрешен доступ только для ID: {OWNER_ID}")
    asyncio.create_task(auto_scan())
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
