"""Flask web app providing a simple chat UI for the FAQ chatbot."""
import ssl
import nltk

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('punkt')
nltk.download('wordnet')

from flask import Flask, render_template, request, jsonify
from chatbot import FAQChatbot

app = Flask(__name__)
bot = FAQChatbot(faq_path="faqs.json", threshold=0.3)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("message", "")
    result = bot.get_response(user_message)
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True)
