from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup


# =====================================================
# КЛАВИАТУРЫ
# =====================================================
def signal_keyboard():
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="📊 Получить сигнал", callback_data="get_best_signal")],
        [InlineKeyboardButton(text="📈 Отчет по сделке", callback_data="trade_report")]
    ])


def trade_report_keyboard():
    return InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(text="✅ Профит", callback_data="report_profit"),
            InlineKeyboardButton(text="❌ Стоп", callback_data="report_stop")
        ],
        [InlineKeyboardButton(text="🔙 Назад", callback_data="back_main")]
    ])