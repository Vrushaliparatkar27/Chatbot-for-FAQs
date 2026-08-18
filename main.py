"""Simple command-line interface for the FAQ chatbot."""

from chatbot import FAQChatbot


def run_cli():
    bot = FAQChatbot(faq_path="faqs.json", threshold=0.3)
    print("FAQ Chatbot (type 'exit' or 'quit' to stop)\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("exit", "quit"):
            print("Bot: Goodbye!")
            break

        result = bot.get_response(user_input)
        print(f"Bot: {result['answer']}")
        if result["matched_question"]:
            print(f"     (matched: \"{result['matched_question']}\" | score: {result['score']:.2f})")
        print()


if __name__ == "__main__":
    run_cli()
