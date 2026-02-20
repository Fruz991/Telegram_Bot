from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup

def signal_keyboard():
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="📊 Получить сигнал", callback_data="get_best_signal")],
        [InlineKeyboardButton(text="📈 Результат сделки", callback_data="trade_result")],
        [InlineKeyboardButton(text="📊 Статистика", callback_data="show_stats")],
    ])

def back_keyboard():
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="⏪ Назад", callback_data="back_main")]
    ])