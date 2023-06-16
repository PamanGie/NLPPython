import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Persiapan data
nltk.download('punkt')
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

# Data pertanyaan dan jawaban
qa_pairs = [
    ("Halo, apa kabar?", "Saya baik-baik saja."),
    ("Apakah kamu ingin makan malam?", "Ya, saya ingin makan malam."),
    ("Apa rencana untuk akhir pekan ini?", "Saya belum membuat rencana spesifik."),
    ("Apa yang kamu pelajari tentang NLP?", "Saya sedang mempelajari pemrosesan bahasa alami (NLP).")
]

# Preprocessing teks
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(lemmatized_tokens)

preprocessed_questions = [preprocess_text(question) for question, _ in qa_pairs]

# Membangun vektor fitur
vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(preprocessed_questions).toarray()

# Fungsi untuk mendapatkan jawaban terbaik berdasarkan pertanyaan pengguna
def get_best_response(user_question):
    preprocessed_input = preprocess_text(user_question)
    input_vector = vectorizer.transform([preprocessed_input]).toarray()

    similarity_scores = cosine_similarity(input_vector, feature_vectors)
    best_index = similarity_scores.argmax()
    _, best_response = qa_pairs[best_index]
    return best_response

# Loop pertanyaan dan jawaban
while True:
    user_question = input("Tanyakan sesuatu (atau ketik 'q' untuk keluar): ")
    if user_question.lower() == 'q':
        break

    best_response = get_best_response(user_question)
    print("AI:", best_response)
