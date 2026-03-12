from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import spacy

from .searcher import SemanticSearcher


class RepresentationAnalyzer:
    """
    Analyze paragraph-level text for semantically relevant representations of
    Indigenous Knowledge using prototype based semantic similarity search.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        threshold: float = 0.75,
    ) -> None:
        """
        Initialize the analyzer.

        Args:
            model_name: Sentence-transformer model used for embedding text.
            threshold: Minimum cosine similarity required to keep a match.
        """
        self.searcher = SemanticSearcher(model_name=model_name)
        self.threshold = threshold
        self.nlp = spacy.load("en_core_web_sm")

        self.boilerplate_patterns = [
            "all rights reserved",
            "copyright",
            "© 20",
            "doi:",
            "published by",
            "journal of",
            "correspondence to",
            "email:",
            "tel:",
            "fax:",
        ]

        self.representation_seeds = {
            "identity_recognition": [
                "indigenous peoples",
                "indigenous communities",
                "first nations communities",
                "aboriginal peoples",
                "knowledge holders",
                "elders and community members",
                "elder",
                "elders",
            ],
            "agency_framing": [
                "indigenous leadership and governance",
                "self-determination rights",
                "indigenous-led initiatives",
                "community control management",
                "indigenous governance systems",
                "community led sampling",
                "community based monitoring",
            ],
            "knowledge_systems": [
                "traditional ecological knowledge",
                "indigenous knowledge systems",
                "indigenous science and expertise",
                "indigenous expertise knowledge",
                "traditional wisdom practices",
                "traditional knowledge",
                "indigenous knowledge",
                "traditional management practices",
                "knowledge exchange",
                "socio-ecological systems",
            ],
            "rights_sovereignty": [
                "indigenous rights sovereignty",
                "land rights territories",
                "free prior informed consent",
                "treaty rights agreements",
                "traditional territories lands",
                "indigenous territory",
                "tribal government",
                "territorial delegations",
                "inuit autonomy",
                "self-governance",
                "accessibility of land",
                "food sovereignty and self-governance",
            ],
            "strength_based": [
                "indigenous resilience adaptation",
                "community strength leadership",
                "indigenous innovation solutions",
                "indigenous leadership success",
                "successful indigenous practices",
            ],
            "deficit_based": [
                "vulnerable indigenous populations",
                "marginalized communities groups",
                "indigenous communities lack",
                "disadvantaged indigenous groups",
                "at-risk indigenous populations",
                "food insecurity",
                "food insecurity assessment",
            ],
            "active_agency": [
                "indigenous peoples manage lead",
                "communities lead initiatives",
                "indigenous control decisions",
                "indigenous peoples practice",
                "communities decide govern",
                "community led sampling",
                "community based monitoring",
            ],
            "passive_positioning": [
                "indigenous peoples affected impacted",
                "communities impacted affected",
                "provided to indigenous",
                "given to communities",
                "indigenous study participants",
            ],
            "cultural_systems": [
                "cultural practice",
                "cultural revitalization",
                "cultural value",
                "cultural responsibility",
                "cultural identity",
                "self-identity",
                "spirituality",
                "cultural property rights",
                "culturally appropriate",
                "traditional diet",
                "traditional foods",
                "food safety",
                "food security",
                "food sovereignty",
                "harvesting",
                "hunter",
                "hunting",
                "safe butchering practices",
                "community freezer",
                "community food storage",
                "inuit rules",
                "inuit laws",
                "inuit practices",
                "inuit protocols",
                "inuit customs",
                "culture harmony and balance",
            ],
            "environmental_relationships": [
                "relationship with the environment",
                "be within the environment",
                "socio-ecological systems",
                "traditional ecological knowledge",
            ],
            "conservation_frameworks": [
                "conservation evaluation",
                "conservation strategy",
                "conservation frameworks",
                "conservation policy framework",
                "conservation policy",
                "holistic frameworks",
                "community based report",
            ],
        }

        self.prototypes = {
            category: self.searcher.compute_prototype(seed_terms)
            for category, seed_terms in self.representation_seeds.items()
        }

    def _is_boilerplate(self, text: str) -> bool:
        """
        Check whether a paragraph appears to contain publication boilerplate
        rather than meaningful analytical content.
        """
        text_lower = text.lower()
        return any(pattern in text_lower for pattern in self.boilerplate_patterns)

    def _split_sentences(self, text: str) -> List[str]:
        """
        Split a paragraph into sentences using spaCy.
        """
        doc = self.nlp(text)
        return [sent.text.strip() for sent in doc.sents if sent.text.strip()]

    def _get_context(
        self,
        sentences: List[str],
        idx: int,
    ) -> Tuple[Optional[str], str, Optional[str]]:
        """
        Return the previous, current, and next sentence for context.

        Args:
            sentences: List of sentences from a paragraph.
            idx: Index of the current sentence.

        Returns:
            A tuple of (previous_sentence, current_sentence, next_sentence).
        """
        previous_sentence = sentences[idx - 1] if idx > 0 else None
        current_sentence = sentences[idx]
        next_sentence = sentences[idx + 1] if idx < len(sentences) - 1 else None
        return previous_sentence, current_sentence, next_sentence

    def analyze_paragraph(self, paragraph: Dict) -> List[Dict]:
        """
        Analyze a paragraph and return all sentence-category matches above
        the similarity threshold.

        Args:
            paragraph: A dictionary containing paragraph text and metadata.

        Returns:
            A list of dictionaries with semantic search results.
        """
        text = paragraph.get("text", "")
        if not text or self._is_boilerplate(text):
            return []

        sentences = self._split_sentences(text)
        if len(sentences) < 3:
            return []

        results: List[Dict] = []
        sentence_embeddings = self.searcher.get_embeddings(sentences).cpu().numpy()

        for idx, (sentence, embedding) in enumerate(zip(sentences, sentence_embeddings)):
            for category, prototype in self.prototypes.items():
                similarity = self.searcher.compute_similarity(embedding, prototype)

                if similarity >= self.threshold:
                    prev_sent, curr_sent, next_sent = self._get_context(sentences, idx)

                    context_parts = [part for part in [prev_sent, curr_sent, next_sent] if part]
                    full_context = " ".join(context_parts)

                    results.append(
                        {
                            "Article": paragraph.get("filename", ""),
                            "Category": category,
                            "Seed_Terms": ", ".join(self.representation_seeds[category]),
                            "Similar_Term": curr_sent,
                            "Similarity_Score": round(similarity, 3),
                            "Sentence": curr_sent,
                            "Full_Context": full_context,
                            "Page_Number": paragraph.get("page_number", ""),
                        }
                    )

        return results