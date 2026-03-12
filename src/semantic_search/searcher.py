from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


class SemanticSearcher:

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        """
        Initialize the semantic search model.

        Args:
            model_name: Name of the sentence-transformer model to load.
        """
        self.model = SentenceTransformer(model_name)

    def get_embeddings(self, texts: Sequence[str]) -> torch.Tensor:
        """
        Encode a list of texts into dense vector representations.

        Args:
            texts: A sequence of input strings.

        Returns:
            A tensor of embeddings.
        """
        return self.model.encode(texts, convert_to_tensor=True)

    def compute_prototype(self, seed_terms: Sequence[str]) -> np.ndarray:
        """
        Create a normalized prototype vector by averaging the embeddings
        of the provided seed terms.

        Args:
            seed_terms: A sequence of seed terms representing a category.

        Returns:
            A normalized NumPy array representing the category prototype.
        """
        embeddings = self.get_embeddings(seed_terms)
        prototype = embeddings.mean(dim=0).cpu().numpy()
        norm = np.linalg.norm(prototype)

        if norm == 0:
            return prototype

        return prototype / norm

    def compute_similarity(
        self,
        text_embedding: np.ndarray,
        prototype: np.ndarray,
    ) -> float:
        """
        Compute cosine similarity between a text embedding and a prototype vector.

        Args:
            text_embedding: Embedding of a sentence or text snippet.
            prototype: Normalized prototype vector for a semantic category.

        Returns:
            Cosine similarity score as a float.
        """
        norm = np.linalg.norm(text_embedding)
        if norm == 0:
            return 0.0

        text_embedding = text_embedding / norm
        return float(np.dot(text_embedding, prototype))