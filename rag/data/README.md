# RAG data

`source_manifest.json` contains a small, curated seed set of authoritative
source summaries and URLs. The full local glyph catalog remains in
`../../data/semantic_meaning.json` and is loaded by `retrieval.py` at runtime.

The JSONL collections (`signs`, `lexicon`, `texts`, `history`, `artifacts`) are
small, human-reviewed seed records and schema examples. They deliberately do
not pretend to be a downloaded copy of the linked sites. For production
ingestion, add JSONL records with a stable `id`, source URL, period, language
stage, Gardiner code, Unicode value, license, and verification date. Manual
review is still required for licensed source acquisition, page-level citation
checking, translations, and refreshing `last_verified`. Do not copy entire
copyrighted books into this folder.

