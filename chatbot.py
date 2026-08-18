"""
Core FAQ Chatbot logic:
- Loads FAQs from JSON
- Preprocesses text using NLTK (tokenize, remove stopwords, lemmatize)
- Matches user query to closest FAQ using TF-IDF + cosine similarity
"""

import json
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def download_nltk_resources():
    resources = ['punkt', 'punkt_tab', 'wordnet']
    for res in resources:
        try:
            nltk.download(res, quiet=True)
        except Exception:
            pass


download_nltk_resources()

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))


def preprocess(text: str) -> str:
    """
    Clean and normalize text:
    - Lowercase
    - Remove punctuation
    - Tokenize
    - Remove stopwords
    - Lemmatize
    """
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = word_tokenize(text)
    tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word not in stop_words and word.isalpha()
    ]
    return " ".join(tokens)


class FAQChatbot:
    def __init__(self, faq_path: str = "faqs.json", threshold: float = 0.3):
        self.threshold = threshold
        self.faqs = self._load_faqs(faq_path)
        self.questions = [faq["question"] for faq in self.faqs]
        self.answers = [faq["answer"] for faq in self.faqs]

        # Preprocess all FAQ questions once at startup
        self.processed_questions = [preprocess(q) for q in self.questions]

        # Fit TF-IDF vectorizer on the FAQ corpus
        self.vectorizer = TfidfVectorizer()
        self.faq_vectors = self.vectorizer.fit_transform(self.processed_questions)

    @staticmethod
    def _load_faqs(path: str) -> list:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def get_response(self, user_query: str) -> dict:
        """
        Returns the best matching FAQ answer along with similarity score.
        If no match crosses the threshold, returns a fallback message.
        """
        if not user_query.strip():
            return {"answer": "Please type a question.", "score": 0.0, "matched_question": None}

        processed_query = preprocess(user_query)
        query_vector = self.vectorizer.transform([processed_query])

        similarities = cosine_similarity(query_vector, self.faq_vectors).flatten()
        best_idx = similarities.argmax()
        best_score = similarities[best_idx]

        if best_score < self.threshold:
            return {
                "answer": "Sorry, I couldn't find a relevant answer. Could you rephrase your question?",
                "score": float(best_score),
                "matched_question": None,
            }

        return {
            "answer": self.answers[best_idx],
            "score": float(best_score),
            "matched_question": self.questions[best_idx],
        }

def get_response(self, user_query):
    if max_similarity < 0.2:
        return "Sorry, I couldn't find a relevant answer. Could you rephrase your question?"
    return best_match_answer