# RAG layer

The RAG layer is part of the chatbot, but it stays in its own package so each
responsibility remains easy to understand:

- `retrieval.py` loads local records and ranks evidence with ChromaDB semantic search plus lexical matching.
- `augmentation.py` builds the grounded prompt and citations.
- `generation_safety.py` validates questions and prepares safe generation.
- `data/` contains curated records, article links, and local books.
- `.chroma/` is the persistent local vector database created at runtime.
- `01_retrieval.ipynb` to `03_generation_safety.ipynb` are small demos.

`chatbot.py` is the application layer. It calls this package, sends the
grounded evidence to Groq, and returns the answer plus sources and similarity.
