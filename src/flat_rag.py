"""Task 7: Flat RAG baseline using TF-IDF + cosine similarity."""
import os, sys
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, os.path.dirname(__file__))
from utils import load_chunks, call_llm

ANSWER_PROMPT = """You are answering a question using only the provided text context.
If the answer is not in the context, say that the information is not available.

Question: {question}

Text context:
{context}

Answer clearly and concisely."""

# Module-level cache
_vectorizer = None
_tfidf_matrix = None
_chunks = None


def build_vector_index(chunks=None):
    """Build TF-IDF index from chunks."""
    global _vectorizer, _tfidf_matrix, _chunks
    if chunks is None:
        chunks = load_chunks()
    _chunks = chunks
    texts = [c["text"] for c in chunks]
    _vectorizer = TfidfVectorizer(stop_words="english")
    _tfidf_matrix = _vectorizer.fit_transform(texts)
    return _vectorizer, _tfidf_matrix


def retrieve_chunks(question, top_k=3):
    """Retrieve top-k most relevant chunks for a question."""
    global _vectorizer, _tfidf_matrix, _chunks
    if _vectorizer is None:
        build_vector_index()
    q_vec = _vectorizer.transform([question])
    sims = cosine_similarity(q_vec, _tfidf_matrix)[0]
    top_indices = np.argsort(sims)[-top_k:][::-1]
    return [_chunks[i] for i in top_indices]


def answer_with_flat_rag(question):
    """Answer question using Flat RAG (TF-IDF retrieval + LLM)."""
    retrieved = retrieve_chunks(question, top_k=3)
    context = "\n\n".join([c["text"] for c in retrieved])
    prompt = ANSWER_PROMPT.format(question=question, context=context)
    return call_llm(prompt)


def main():
    build_vector_index()
    questions = [
        "Who founded OpenAI and what products is it known for?",
        "What is the relationship between Microsoft and OpenAI?",
        "Which companies did Elon Musk found?",
    ]
    for q in questions:
        print(f"\nQ: {q}")
        ans = answer_with_flat_rag(q)
        print(f"A: {ans}")


if __name__ == "__main__":
    main()
