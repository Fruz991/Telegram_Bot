import asyncio
import os
import time

from aiogram import Bot, Dispatcher, types, F
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.filters import Command
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton
from dotenv import load_dotenv

from config import SYMBOLS, COOLDOWN_SECONDS
from tracker import tracker
from keyboards import signal_keyboard
from analytics import analyze_all_timeframes_async, format_signal

# =====================================================
# ЗАГРУЗКА ТОКЕНОВ
# =====================================================
load_dotenv()

TOKEN = os.getenv("BOT_TOKEN")
OWNER_ID = int(os.getenv("OWNER_ID"))

if not TOKEN:
    exit("Ошибка: BOT_TOKEN не найден в .env")
if not OWNER_ID:
    exit("Ошибка: OWNER_ID не найден в .env")

# =====================================================
# БОТ
# =====================================================
session = AiohttpSession(timeout=10)
bot = Bot(token=TOKEN, session=session)
dp = Dispatcher()

signal_cooldown = {}
pending_signal = {}  # Хранит найденный сигнал до нажатия кнопки


# =====================================================
# ПРОВЕРКА ДОСТУПА
# =====================================================
async def check_access_message(message: types.Message) -> bool:
    if message.from_user.id != OWNER_ID:
        await message.reply("⚠️ На данный момент бот находится в Техническом Обслуживании")
        return False
    return True


async def check_access_callback(callback: types.CallbackQuery) -> bool:
    if callback.from_user.id != OWNER_ID:
        await callback.answer("⚠️ Бот на обслуживании", show_alert=True)
        return False
    return True


# =====================================================
# КЛАВИАТУРА УВЕДОМЛЕНИЯ О СИГНАЛЕ
# =====================================================
def alert_keyboard():
    """Кнопка появляется когда бот нашёл сигнал"""
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="📊 Получить сигнал", callback_data="get_pending_signal")],
        [InlineKeyboardButton(text="❌ Пропустить", callback_data="skip_signal")]
    ])


# =====================================================
# КОМАНДЫ
# =====================================================
@dp.message(Command("start"))
async def send_start(message: types.Message):
    if not await check_access_message(message):
        return

    stops_count = tracker.get_stops_count()
    await message.reply(
        f"👋 Привет! Стопов сегодня: {stops_count}/3\n\nБот анализирует рынок 24/7 и уведомит тебя когда появится возможность для входа.\n\nИли нажми кнопку чтобы найти сигнал прямо сейчас:",
        reply_markup=signal_keyboard()
    )


@dp.message(Command("signal"))
async def send_signal(message: types.Message):
    if not await check_access_message(message):
        return

    if not tracker.can_trade():
        await message.reply("🚫 Лимит на сегодня достигнут. Иди отдыхай.")
        return

    await message.reply("🔍 Анализирую рынок...")

    now = time.time()
    for symbol in SYMBOLS:
        if symbol in signal_cooldown:
            if now - signal_cooldown[symbol] < COOLDOWN_SECONDS:
                continue

        signal = await analyze_all_timeframes_async(symbol)
        if signal['side'] != "NO SIGNAL":
            signal_cooldown[symbol] = now
            await message.reply(format_signal(signal))
            return

    await message.reply("⏳ Сигналов сейчас нет. Бот продолжает мониторинг 24/7.")


@dp.message(Command("stats"))
async def show_stats(message: types.Message):
    if not await check_access_message(message):
        return

    stops_count = tracker.get_stops_count()
    status = "✅ Можно торговать" if tracker.can_trade() else "🚫 Лимит достигнут"

    await message.reply(f"""
📊 Статистика на сегодня:
Стопов: {stops_count}/3
Статус: {status}
""")


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
    for symbol in SYMBOLS:
        if symbol in signal_cooldown:
            if now - signal_cooldown[symbol] < COOLDOWN_SECONDS:
                continue

        signal = await analyze_all_timeframes_async(symbol)
        if signal['side'] != "NO SIGNAL":
            signal_cooldown[symbol] = now
            await callback.message.answer(format_signal(signal))
            return

    await callback.message.answer("⏳ Сейчас нет сильных сигналов. Бот продолжает мониторинг 24/7.")


@dp.callback_query(F.data == "get_pending_signal")
async def get_pending_signal(callback: types.CallbackQuery):
    """Выдаёт сигнал который нашёл автоскан"""
    if not await check_access_callback(callback):
        return

    signal = pending_signal.get("signal")

    if not signal:
        await callback.answer("⚠️ Сигнал уже устарел. Жди нового уведомления.", show_alert=True)
        return

    # Очищаем pending сигнал
    pending_signal.clear()

    await callback.message.answer(format_signal(signal))
    await callback.answer()


@dp.callback_query(F.data == "skip_signal")
async def skip_signal(callback: types.CallbackQuery):
    """Пропустить найденный сигнал"""
    if not await check_access_callback(callback):
        return

    pending_signal.clear()
    await callback.message.edit_text("❌ Сигнал пропущен. Продолжаю мониторинг...")
    await callback.answer()


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
    while True:
        try:
            if not tracker.can_trade():
                await asyncio.sleep(3600)
                continue

            now = time.time()
            for symbol in SYMBOLS:
                if symbol in signal_cooldown:
                    if now - signal_cooldown[symbol] < COOLDOWN_SECONDS:
                        continue

                signal = await analyze_all_timeframes_async(symbol)

                if signal['side'] != "NO SIGNAL":
                    signal_cooldown[symbol] = now

                    # Сохраняем сигнал и отправляем уведомление с кнопкой
                    pending_signal["signal"] = signal

                    symbol_fmt = signal['symbol'].replace('/', '')
                    side = signal['side']
                    emoji = "📈" if side == "LONG" else "📉"

                    await bot.send_message(
                        OWNER_ID,
                        f"🔔 Появилась возможность для входа в позицию!\n\n"
                        f"Монета: {symbol_fmt} {emoji}\n"
                        f"Направление: {side}\n\n"
                        f"Нажми кнопку чтобы получить полный сигнал 👇",
                        reply_markup=alert_keyboard()
                    )
                    # Небольшая пауза после нахождения сигнала
                    await asyncio.sleep(60)

        except Exception as e:
            print(f"Ошибка автоскана: {e}")

        await asyncio.sleep(300)  # Скан каждые 5 минут


# =====================================================
# ЗАПУСК
# =====================================================
async def main():
    print("Бот запущен и ждёт сообщений...")
    print(f"Разрешен доступ только для ID: {OWNER_ID}")
    task = asyncio.create_task(auto_scan())
    try:
        await dp.start_polling(bot)
    finally:
        task.cancel()


if __name__ == "__main__":
    asyncio.run(main())