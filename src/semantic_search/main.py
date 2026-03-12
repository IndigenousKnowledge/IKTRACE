from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

from tqdm import tqdm

from .analyzer import RepresentationAnalyzer


OUTPUT_FIELDS = [
    "Article",
    "Category",
    "Seed_Terms",
    "Similar_Term",
    "Similarity_Score",
    "Sentence",
    "Full_Context",
    "Page_Number",
]


def iter_json_files(input_dir: Path) -> list[Path]:
    """
    Return all JSON files in the input directory in sorted order.
    """
    return sorted(input_dir.glob("*.json"))


def process_json_files(
    input_dir: Path,
    output_csv: Path,
    threshold: float,
    model_name: str,
) -> None:
    """
    Run semantic representation analysis over all JSON files in a directory.

    Args:
        input_dir: Directory containing processed JSON files.
        output_csv: Path to the output CSV file.
        threshold: Similarity threshold for retaining matches.
        model_name: Sentence-transformer model name.
    """
    analyzer = RepresentationAnalyzer(
        model_name=model_name,
        threshold=threshold,
    )

    all_results = []
    json_files = iter_json_files(input_dir)

    logging.info("Found %d JSON files in %s", len(json_files), input_dir)

    for json_file in tqdm(json_files, desc="Processing files"):
        try:
            with json_file.open("r", encoding="utf-8") as f:
                paragraphs = json.load(f)

            for paragraph in paragraphs:
                results = analyzer.analyze_paragraph(paragraph)
                all_results.extend(results)

        except json.JSONDecodeError:
            logging.warning("Skipped invalid JSON file: %s", json_file.name)
        except Exception as exc:
            logging.warning("Skipped file %s due to error: %s", json_file.name, exc)

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(all_results)

    logging.info("Total matches found: %d", len(all_results))
    logging.info("Results saved to %s", output_csv)


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run semantic similarity analysis over processed JSON files."
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        required=True,
        help="Directory containing processed JSON files.",
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=Path("representation_analysis.csv"),
        help="Path to the output CSV file.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Similarity threshold for keeping matches.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence-transformer model name.",
    )
    return parser.parse_args()


def main() -> None:
    """
    Entry point for command-line execution.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()

    process_json_files(
        input_dir=args.input_dir,
        output_csv=args.output_csv,
        threshold=args.threshold,
        model_name=args.model_name,
    )


if __name__ == "__main__":
    main()