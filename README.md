# IKTRACE  
## A Natural Language Processing Framework for Identifying Indigenous Knowledge Representation in Academic Literature  

This repository contains the implementation and research artifacts for the paper:  
**IKTRACE: A Natural Language Processing Framework for Identifying Indigenous Knowledge Representation in Academic Literature.**  

The project introduces a community-informed taxonomy and a multi-method NLP pipeline for identifying Indigenous Knowledge (IK) representations in academic literature while respecting Indigenous Data Sovereignty principles.  

**Raw source documents are not included due to copyright and Indigenous Data Sovereignty considerations.**  
**The LLM-based extraction component described in the paper (LlamaExtract) is not included, as outputs require further human validation.**  

---

## Indigenous Data Sovereignty and Ethical Use  

This project follows Indigenous Data Sovereignty principles and community-engaged research practices.  

Users of this repository must:  

- Respect Indigenous intellectual sovereignty  
- Avoid misuse or misrepresentation of Indigenous Knowledge  
- Cite the associated publication appropriately  
- Refrain from commercial reuse of derived knowledge representations without community engagement  

Some materials are intentionally excluded to ensure responsible data stewardship.  

---

## Repository Contents (Brief)

This repository includes:  

- Processed paragraph-level JSON files  
- Indigenous Knowledge taxonomy domains  
- GLiNER entity extraction outputs  
- Semantic similarity analysis outputs  
- KeyBERT keyword extraction outputs  
- Knowledge graph export files  
- NLP pipeline implementation code  

---

## Repository Structure  

IKTRACE/
├── README.md
├── requirements.txt
├── LICENSE
│
├── data/
│   ├── processed_json/
│   ├── taxonomy/
│   └── knowledge_graph/
│
├── outputs/
│   ├── gliner/
│   ├── semantic_search/
│   └── keybert/
│
└── src/
    ├── gliner/
    ├── semantic_search/
    └── keybert/

