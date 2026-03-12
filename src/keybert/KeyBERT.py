from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
from keybert import KeyBERT
from sklearn.metrics.pairwise import cosine_similarity


def parse_target_terms(target_terms_str: str) -> list[str]:
    """Split a comma-separated target-term string into a clean list."""
    return [term.strip() for term in target_terms_str.split(",") if term.strip()]


def get_best_match(
    text: str,
    target_terms_str: str,
    kw_model: KeyBERT,
    top_n: int = 10,
) -> Optional[str]:
    """
    Extract candidate keywords from a text snippet and return the keyword
    whose embedding is most similar to the provided target terms.
    """
    if pd.isna(text) or pd.isna(target_terms_str):
        return None

    target_terms = parse_target_terms(target_terms_str)
    if not target_terms:
        return None

    keywords = kw_model.extract_keywords(
        text,
        keyphrase_ngram_range=(1, 3),
        top_n=top_n,
    )

    valid_keywords = [
        (keyword, score)
        for keyword, score in keywords
        if keyword.lower() in text.lower()
    ]

    if not valid_keywords:
        return None

    target_embeddings = kw_model.model.embed(target_terms)
    keyword_texts = [keyword for keyword, _ in valid_keywords]
    keyword_embeddings = kw_model.model.embed(keyword_texts)

    best_keyword = None
    best_similarity = -1.0

    for i, keyword in enumerate(keyword_texts):
        keyword_embedding = keyword_embeddings[i : i + 1]
        similarities = cosine_similarity(keyword_embedding, target_embeddings)[0]
        max_similarity = similarities.max()

        if max_similarity > best_similarity:
            best_similarity = max_similarity
            best_keyword = keyword

    return best_keyword


def process_file(
    input_file: Path,
    output_file: Path,
    text_column: str,
    target_terms_column: str,
    output_column: str,
    top_n: int,
) -> None:
    """Load the input spreadsheet, extract best matching keywords, and save results."""
    logging.info("Loading input file: %s", input_file)
    df = pd.read_excel(input_file)

    logging.info("Initializing KeyBERT model...")
    kw_model = KeyBERT()

    logging.info("Processing %d rows...", len(df))
    df[output_column] = df.apply(
        lambda row: get_best_match(
            text=row[text_column],
            target_terms_str=row[target_terms_column],
            kw_model=kw_model,
            top_n=top_n,
        ),
        axis=1,
    )

    df.to_excel(output_file, index=False)
    logging.info("Saved results to %s", output_file)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract best matching KeyBERT terms from semantic similarity outputs."
    )
    parser.add_argument(
        "--input_file",
        type=Path,
        required=True,
        help="Path to the input Excel file.",
    )
    parser.add_argument(
        "--output_file",
        type=Path,
        required=True,
        help="Path to the output Excel file.",
    )
    parser.add_argument(
        "--text_column",
        type=str,
        default="Similar_Text",
        help="Column containing the text snippets.",
    )
    parser.add_argument(
        "--target_terms_column",
        type=str,
        default="Seed_Terms",
        help="Column containing the comma-separated target terms.",
    )
    parser.add_argument(
        "--output_column",
        type=str,
        default="Extracted_Term",
        help="Name of the output column for extracted terms.",
    )
    parser.add_argument(
        "--top_n",
        type=int,
        default=10,
        help="Number of candidate keywords to extract per text snippet.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()

    process_file(
        input_file=args.input_file,
        output_file=args.output_file,
        text_column=args.text_column,
        target_terms_column=args.target_terms_column,
        output_column=args.output_column,
        top_n=args.top_n,
    )


if __name__ == "__main__":
    main()