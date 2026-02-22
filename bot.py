import asyncio
import os
import time
import logging
from aiogram import Bot, Dispatcher, types, F
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.filters import Command
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton
from dotenv import load_dotenv
from config import SYMBOLS, COOLDOWN_SECONDS, SCAN_INTERVAL_SECONDS
from tracker import tracker
from keyboards import signal_keyboard
from analytics import analyze_all_timeframes_async, format_signal, get_market_context_cached


# =====================================================
# ЛОГИРОВАНИЕ
# =====================================================
logging.basicConfig(
    level=logging.INFO,  # INFO — тільки важливі повідомлення
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bot.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# Вимикаємо DEBUG для HTTP-запитів (щоб не спамило)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('ccxt').setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# =====================================================
# ЗАГРУЗКА ТОКЕНОВ
# =====================================================
load_dotenv()
TOKEN = os.getenv("BOT_TOKEN")
OWNER_ID = int(os.getenv("OWNER_ID"))

if not TOKEN:
    logger.error("Ошибка: BOT_TOKEN не найден в .env")
    exit("Ошибка: BOT_TOKEN не найден в .env")
if not OWNER_ID:
    logger.error("Ошибка: OWNER_ID не найден в .env")
    exit("Ошибка: OWNER_ID не найден в .env")

logger.info(f"Бот запускается. Owner ID: {OWNER_ID}")

# =====================================================
# БОТ
# =====================================================
session = AiohttpSession(timeout=10)
bot = Bot(token=TOKEN, session=session)
dp = Dispatcher()
signal_cooldown = {}
pending_signal = {}
last_signals = {}  # Хранение последних сигналов: {symbol: {"side": "...", "time": timestamp}}

# =====================================================
# ПРОВЕРКА НА ДУПЛИКАТЫ СИГНАЛОВ
# =====================================================
SIGNAL_DUPLICATE_WINDOW = 3600  # 1 час - не показывать тот же сигнал по той же монете

def is_duplicate_signal(symbol: str, side: str) -> bool:
    """Проверяет, не был ли уже отправлен такой же сигнал недавно"""
    now = time.time()
    if symbol in last_signals:
        last_signal = last_signals[symbol]
        # Если тот же direction и не прошло достаточно времени
        if last_signal["side"] == side:
            if now - last_signal["time"] < SIGNAL_DUPLICATE_WINDOW:
                return True
    return False

def update_last_signal(symbol: str, side: str):
    """Обновляет время последнего сигнала по монете"""
    last_signals[symbol] = {"side": side, "time": time.time()}

# =====================================================
# ПРОВЕРКА ДОСТУПА
# =====================================================
async def check_access_message(message: types.Message) -> bool:
    if message.from_user.id != OWNER_ID:
        await message.reply("⚠️ На данный момент бот находится в Техническом Обслуживании")
        logger.warning(f"Попытка доступа от {message.from_user.id}")
        return False
    return True

async def check_access_callback(callback: types.CallbackQuery) -> bool:
    if callback.from_user.id != OWNER_ID:
        await callback.answer("⚠️ Бот на обслуживании", show_alert=True)
        return False
    return True

# =====================================================
# КЛАВИАТУРА УВЕДОМЛЕНИЯ
# =====================================================
def alert_keyboard():
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="📊 Получить сигнал", callback_data="get_pending_signal")],
        [InlineKeyboardButton(text="❌ Пропустить", callback_data="skip_signal")]
    ])

def trade_result_keyboard():
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="✅ Win", callback_data="result_win")],
        [InlineKeyboardButton(text="❌ Loss", callback_data="result_loss")],
    ])

def back_keyboard():
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="⏪ Назад", callback_data="back_main")]
    ])

# =====================================================
# КОМАНДЫ
# =====================================================
@dp.message(Command("start"))
async def send_start(message: types.Message):
    if not await check_access_message(message):
        return
    stops_count = tracker.get_stops_count()
    wins_count = tracker.get_wins_count()
    await message.reply(
        f"👋 Привет! Стопов сегодня: {stops_count}/3\n"
        f"✅ Винсов сегодня: {wins_count}\n\n"
        f"Бот анализирует рынок 24/7 и уведомит тебя когда появится возможность для входа.\n"
        f"Или нажми кнопку чтобы найти сигнал прямо сейчас:",
        reply_markup=signal_keyboard()
    )
    logger.info("Команда /start выполнена")

@dp.message(Command("signal"))
async def send_signal(message: types.Message):
    if not await check_access_message(message):
        return
    if not tracker.can_trade():
        await message.reply("🚫 Лимит на сегодня достигнут. Иди отдыхай.")
        return

    await message.reply("🔍 Анализирую рынок...")

    now = time.time()
    market_context = await get_market_context_cached()

    for symbol in SYMBOLS:
        if symbol in signal_cooldown:
            if now - signal_cooldown[symbol] < COOLDOWN_SECONDS:
                continue

        signal = await analyze_all_timeframes_async(symbol, market_context)
        if signal['side'] != "NO SIGNAL":
            # Проверка на дубликат
            if is_duplicate_signal(symbol, signal['side']):
                logger.info(f"Сигнал {symbol} {signal['side']} пропущен (дубликат)")
                signal_cooldown[symbol] = now
                continue
            
            signal_cooldown[symbol] = now
            update_last_signal(symbol, signal['side'])
            await message.reply(format_signal(signal))
            logger.info(f"Сигнал найден: {symbol} {signal['side']}")
            return

    await message.reply("⏳ Сигналов сейчас нет. Бот продолжает мониторинг 24/7.")
    logger.debug("Сигналов не найдено по команде /signal")

@dp.message(Command("stats"))
async def show_stats(message: types.Message):
    if not await check_access_message(message):
        return
    stats = tracker.get_stats()
    await message.reply(f"""
📊 Статистика на сегодня:
Стопов: {stats['daily_stops']}/3
Винсов: {stats['daily_wins']}
Всего сделок: {stats['total_trades']}
Win Rate: {stats['win_rate']}%
Статус: {'✅ Можно торговать' if stats['can_trade'] else '🚫 Лимит достигнут'}
""")
    logger.info("Показана статистика")

@dp.message(Command("result"))
async def trade_result_menu(message: types.Message):
    if not await check_access_message(message):
        return
    await message.reply("Выберите результат сделки:", reply_markup=trade_result_keyboard())
    logger.info("Открыто меню результата сделки")

# =====================================================
# CALLBACKS
# =====================================================
@dp.callback_query(F.data == "get_best_signal")
async def send_best_signal(callback: types.CallbackQuery):
    if not await check_access_callback(callback):
        return
    if not tracker.can_trade():
        await callback.answer("🚫 Лимит на сегодня достигнут. Иди отдыхай.", show_alert=True)
        return

    await callback.answer("🔍 Анализирую рынок...")

    now = time.time()
    market_context = await get_market_context_cached()

    for symbol in SYMBOLS:
        if symbol in signal_cooldown:
            if now - signal_cooldown[symbol] < COOLDOWN_SECONDS:
                continue

        signal = await analyze_all_timeframes_async(symbol, market_context)
        if signal['side'] != "NO SIGNAL":
            # Проверка на дубликат
            if is_duplicate_signal(symbol, signal['side']):
                logger.info(f"Сигнал {symbol} {signal['side']} пропущен (дубликат)")
                signal_cooldown[symbol] = now
                continue
            
            signal_cooldown[symbol] = now
            update_last_signal(symbol, signal['side'])
            await callback.message.answer(format_signal(signal))
            logger.info(f"Сигнал найден через callback: {symbol}")
            return

    await callback.message.answer("⏳ Сейчас нет сильных сигналов. Бот продолжает мониторинг 24/7.")

@dp.callback_query(F.data == "get_pending_signal")
async def get_pending_signal(callback: types.CallbackQuery):
    if not await check_access_callback(callback):
        return
    signal = pending_signal.get("signal")
    if not signal:
        await callback.answer("⚠️ Сигнал уже устарел. Жди нового уведомления.", show_alert=True)
        return
    
    pending_signal.clear()
    await callback.message.answer(format_signal(signal))
    await callback.answer()
    logger.info("Пользователь получил pending сигнал")

@dp.callback_query(F.data == "skip_signal")
async def skip_signal(callback: types.CallbackQuery):
    if not await check_access_callback(callback):
        return
    pending_signal.clear()
    await callback.message.edit_text("❌ Сигнал пропущен. Продолжаю мониторинг...")
    await callback.answer()
    logger.info("Сигнал пропущен пользователем")

@dp.callback_query(F.data == "result_win")
async def result_win(callback: types.CallbackQuery):
    if not await check_access_callback(callback):
        return
    tracker.add_win()
    await callback.message.answer("✅ Вин записан! Так держать! 🚀", reply_markup=back_keyboard())
    await callback.answer()
    logger.info("Записан WIN")

@dp.callback_query(F.data == "result_loss")
async def result_loss(callback: types.CallbackQuery):
    if not await check_access_callback(callback):
        return
    tracker.add_stop()
    stops = tracker.get_stops_count()
    await callback.message.answer(f"❌ Стоп записан. Сегодня стопов: {stops}/3", reply_markup=back_keyboard())
    await callback.answer()
    logger.info(f"Записан LOSS. Всего: {stops}")

@dp.callback_query(F.data == "back_main")
async def back_to_main(callback: types.CallbackQuery):
    if not await check_access_callback(callback):
        return
    await callback.message.answer("Главное меню:", reply_markup=signal_keyboard())
    await callback.answer()

# =====================================================
# АВТОСКАН 24/7
# =====================================================
async def auto_scan():
    logger.info("Авто-скан запущен")
    while True:
        try:
            if not tracker.can_trade():
                logger.warning("Лимит стопов достигнут, пауза 1 час")
                await asyncio.sleep(3600)
                continue

            now = time.time()
            signals_found = 0
            market_context = await get_market_context_cached()

            btc_context = market_context["btc"]
            logger.info(f"BTC контекст: {btc_context}")

            for symbol in SYMBOLS:
                if symbol in signal_cooldown:
                    if now - signal_cooldown[symbol] < COOLDOWN_SECONDS:
                        continue

                signal = await analyze_all_timeframes_async(symbol, market_context)

                if signal['side'] != "NO SIGNAL":
                    # Проверка на дубликат
                    if is_duplicate_signal(symbol, signal['side']):
                        logger.info(f"Сигнал {symbol} {signal['side']} пропущен (дубликат)")
                        signal_cooldown[symbol] = now
                        continue
                    
                    signal_cooldown[symbol] = now
                    update_last_signal(symbol, signal['side'])
                    pending_signal["signal"] = signal
                    symbol_fmt = signal['symbol'].replace('/', '')
                    side = signal['side']
                    emoji = "📈" if side == "LONG" else "📉"

                    await bot.send_message(
                        OWNER_ID,
                        f"🔔 Появилась возможность для входа в позицию!\n"
                        f"Монета: {symbol_fmt} {emoji}\n"
                        f"Направление: {side}\n"
                        f"Нажми кнопку чтобы получить полный сигнал 👇",
                        reply_markup=alert_keyboard()
                    )
                    signals_found += 1
                    logger.info(f"Сигнал найден: {symbol_fmt} {side}")
                    await asyncio.sleep(5)

            if signals_found == 0:
                logger.debug("Сигналов не найдено в этом цикле")

            await asyncio.sleep(SCAN_INTERVAL_SECONDS)

        except Exception as e:
            logger.error(f"Ошибка автоскана: {e}")
            await asyncio.sleep(SCAN_INTERVAL_SECONDS)

# =====================================================
# ЗАПУСК
# =====================================================
async def main():
    logger.info("Бот запущен и ждёт сообщений...")
    logger.info(f"Разрешен доступ только для ID: {OWNER_ID}")
    task = asyncio.create_task(auto_scan())
    try:
        await dp.start_polling(bot)
    finally:
        task.cancel()
        await bot.session.close()
        logger.info("Бот остановлен")

if __name__ == "__main__":
    asyncio.run(main())