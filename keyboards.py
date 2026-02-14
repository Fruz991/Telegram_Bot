from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup


# =====================================================
# ГЛАВНАЯ КЛАВИАТУРА
# =====================================================
def signal_keyboard():
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="📊 Получить сигнал", callback_data="get_best_signal")],
    ])