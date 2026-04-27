"""Utility functions for text similarity and other shared helpers."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.paper import Paper


def compute_similarity_matrix(papers: list[Paper], threshold: float = 0.15) -> list[tuple[str, str, float]]:
    """Compute pairwise cosine similarity between paper texts using TF-IDF.

    Returns a list of (paper_id_a, paper_id_b, similarity_score) tuples
    for all pairs exceeding the given threshold.
    """
    if len(papers) < 2:
        return []

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    texts = [p.similarity_features() for p in papers]
    ids = [p.paper_id for p in papers]

    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    try:
        tfidf_matrix = vectorizer.fit_transform(texts)
    except ValueError:
        # All documents contain only stop words or are empty
        return []
    sim_matrix = cosine_similarity(tfidf_matrix)

    edges: list[tuple[str, str, float]] = []
    for i in range(len(papers)):
        for j in range(i + 1, len(papers)):
            score = float(sim_matrix[i, j])
            if score >= threshold:
                edges.append((ids[i], ids[j], round(score, 4)))

    return edges


def compute_pairwise_similarity(paper_a: Paper, paper_b: Paper) -> float:
    """Compute TF-IDF cosine similarity between exactly two papers.

    Returns a float in [0, 1].  Returns 0.0 if the texts produce no
    TF-IDF features (e.g. only stop words).
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    texts = [paper_a.similarity_features(), paper_b.similarity_features()]
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    try:
        tfidf_matrix = vectorizer.fit_transform(texts)
    except ValueError:
        return 0.0
    sim = cosine_similarity(tfidf_matrix)
    return float(round(sim[0, 1], 4))


def clean_text(text: str) -> str:
    """Remove excess whitespace and normalize a text string."""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def truncate(text: str, max_length: int = 200) -> str:
    """Truncate text to a maximum length with an ellipsis."""
    if len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."
