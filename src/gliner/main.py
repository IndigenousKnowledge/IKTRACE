from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Iterable, Tuple

import pandas as pd
import spacy
from gliner import GLiNER


DEFAULT_LABELS = [
    "elder",
    "Indigenous knowledge holder",
    "community scholar",
    "Indigenous organization",
    "government agency",
    "co-management body",
    "Inuit Circumpolar Council",
    "Government of Nunavut",
    "Parks Canada",
    "Makivvik Corporation",
    "Nunavut Research Institute",
    "Indigenous community",
    "Inuit community",
    "Nunavut",
    "Nunavik",
    "Inuit Nunangat",
    "protected area",
    "Inuktitut",
    "traditional knowledge",
    "food security",
]


def clean_text(text: str) -> str:
    """Normalize whitespace in a paragraph."""
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def get_sentence_context(
    nlp, entity_text: str, full_text: str
) -> Tuple[str, str]:
    """
    Return the sentence containing the entity and the context window
    (previous, current, next sentences).
    """
    doc = nlp(full_text)
    sentences = list(doc.sents)

    for i, sent in enumerate(sentences):
        if entity_text.lower() in sent.text.lower():
            previous_sent = sentences[i - 1].text.strip() if i > 0 else ""
            current_sent = sent.text.strip()
            next_sent = sentences[i + 1].text.strip() if i < len(sentences) - 1 else ""

            context = " ".join(filter(None, [previous_sent, current_sent, next_sent]))
            return current_sent, context

    return "", ""


def is_citation(text: str) -> bool:
    """detect likely citation/reference-like paragraphs."""
    text = text.strip()

    name_list_pattern = (
        r"[A-Z][a-z]+\s+[A-Z]\.\s+[A-Z]\.,\s*[A-Z][a-z]+\s+[A-Z]\.\s+[A-Z]\."
    )
    if re.search(name_list_pattern, text):
        return True

    author_pattern = re.match(r"^\s*[A-Z][a-z]+\s+[A-Z]{1,2}", text)
    if author_pattern:
        has_et_al = re.search(r"et\s+al\.", text)
        has_publication_info = re.search(
            r"(PMID:|doi:|Journal|PLoS|\d{4};\s*\d+:)", text
        )
        if has_et_al or has_publication_info:
            return True

    return False


def has_min_content(text: str, min_words: int = 45) -> bool:
    """Check whether a paragraph meets the minimum word-count threshold."""
    return len(text.split()) >= min_words


def iter_json_files(input_dir: Path) -> Iterable[Path]:
    """Yield JSON files from the input directory."""
    return sorted(input_dir.glob("*.json"))


def extract_entities_from_paragraphs(
    paragraphs: list[dict],
    model: GLiNER,
    nlp,
    labels: list[str],
    threshold: float,
    min_words: int,
) -> list[dict]:
    """Run entity extraction over a list of paragraph dictionaries."""
    results = []

    for para in paragraphs:
        raw_text = para.get("text", "")
        text = clean_text(raw_text)

        if not text or is_citation(text) or not has_min_content(text, min_words=min_words):
            continue

        entities = model.predict_entities(text, labels, threshold=threshold)

        for entity in entities:
            sentence, context = get_sentence_context(nlp, entity["text"], text)
            results.append(
                {
                    "article_name": para.get("filename", ""),
                    "page_number": para.get("page_number", ""),
                    "label": entity["label"],
                    "term": entity["text"],
                    "confidence": entity["score"],
                    "sentence": sentence,
                    "context": context,
                }
            )

    return results


def run_extraction(
    input_dir: Path,
    output_file: Path,
    model_name: str,
    threshold: float,
    min_words: int,
) -> None:
    """Run GLiNER extraction over all JSON files in a directory."""
    logging.info("Loading spaCy model...")
    nlp = spacy.load("en_core_web_sm")

    logging.info("Loading GLiNER model: %s", model_name)
    model = GLiNER.from_pretrained(model_name)
    model.eval()

    all_results = []
    json_files = list(iter_json_files(input_dir))

    logging.info("Found %d JSON files in %s", len(json_files), input_dir)

    for json_file in json_files:
        logging.info("Processing %s", json_file.name)

        with json_file.open("r", encoding="utf-8") as f:
            paragraphs = json.load(f)

        file_results = extract_entities_from_paragraphs(
            paragraphs=paragraphs,
            model=model,
            nlp=nlp,
            labels=DEFAULT_LABELS,
            threshold=threshold,
            min_words=min_words,
        )
        all_results.extend(file_results)

    df = pd.DataFrame(all_results)
    df.to_csv(output_file, index=False)

    logging.info("Total entities found: %d", len(df))
    if not df.empty:
        logging.info("Entities by type:\n%s", df["label"].value_counts())
    logging.info("Saved results to %s", output_file)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract Indigenous Knowledge-related entities from processed JSON files using GLiNER."
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        required=True,
        help="Path to the folder containing processed JSON files.",
    )
    parser.add_argument(
        "--output_file",
        type=Path,
        default=Path("extracted_entities.csv"),
        help="Path to the output CSV file.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="urchade/gliner_medium",
        help="Hugging Face model name for GLiNER.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.4,
        help="Confidence threshold for entity prediction.",
    )
    parser.add_argument(
        "--min_words",
        type=int,
        default=45,
        help="Minimum number of words required to process a paragraph.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()

    run_extraction(
        input_dir=args.input_dir,
        output_file=args.output_file,
        model_name=args.model_name,
        threshold=args.threshold,
        min_words=args.min_words,
    )


if __name__ == "__main__":
    main()