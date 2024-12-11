from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)
from dotenv import load_dotenv
import os
from src.llm.llm import answer_question
import warnings

warnings.filterwarnings("ignore")

load_dotenv()
# Replace 'YOUR_BOT_TOKEN' with your actual Telegram bot token


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        f"Merhaba ben GeoBot, size nasıl yardımcı olabilirim?"
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        f"TBDY 2018 hakkında bilgi almak için /custom komutunu kullanın."
    )


async def custom_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text: str = update.message.text

    response = answer_question(text)
    await update.message.reply_text(response)


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message_type: str = update.message.chat.type
    text: str = update.message.text

    print(
        f"User {update.effective_user.first_name} sent a message of type {message_type} with text {text}"
    )

    if text == "/start":
        await start_command(update, context)
    elif text == "/help":
        await help_command(update, context)
    else:
        await custom_command(update, context)


async def error(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Log Errors caused by Updates."""
    print(f"Update {update} caused error {context.error}")


def main():
    """Start the bot."""
    # Create an updater and pass in the bot token
    print("Starting the bot")
    token = os.getenv("TELEGRAM_TOKEN", "")
    app = ApplicationBuilder().token(token).build()

    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("custom", custom_command))

    app.add_handler(MessageHandler(filters.TEXT, handle_message))

    app.add_error_handler(error)

    print("Polling")
    app.run_polling(poll_interval=3)


if __name__ == "__main__":
    main()
