from dotenv import load_dotenv
import telebot

import os

from classify_text.service import ClassifyText

load_dotenv()

TOKEN = os.environ.get("BOT_TOKEN")
bot = telebot.TeleBot(TOKEN)

@bot.message_handler(content_types=["document"])
def handle_txt_file(message):
    file_name = message.document.file_name
    if not file_name.endswith(".txt"):
        bot.reply_to(message, "❌ Пожалуйста, отправьте только `.txt` файл.")
        return

    try:
        file_info = bot.get_file(message.document.file_id)
        downloaded_file = bot.download_file(file_info.file_path)

        temp_path = "temp_text.txt"
        with open(temp_path, "wb") as f:
            f.write(downloaded_file)

        classifier = ClassifyText(temp_path)
        res = classifier.classify_text()

        bot.reply_to(message, f"📂 Категория текста: *{res}*", parse_mode="Markdown")

        os.remove(temp_path)

    except Exception as e:
        bot.reply_to(message, f"❌ Ошибка обработки: {e}")

@bot.message_handler(commands=["start", "help"])
def send_welcome(message):
    bot.reply_to(message, "👋 Привет! Пришли мне `.txt` файл с текстом, и я скажу, к какой рубрике он относится.")


if __name__ == "__main__":
    print("🤖 Бот запущен.")
    bot.polling()
        
