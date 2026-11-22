# Kemet Studio

Kemet Studio is a FastAPI application for Ancient Egyptian hieroglyphs. It
combines a grounded RAG chatbot, image detection, a sign catalog, and speech.

## Start

```powershell
pip install -r requirements-chatbot.txt
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000>.

Create a local `.env` file with:

```env
GROQ_API_KEY=your_key_here
GROQ_MODEL=openai/gpt-oss-20b
GLYPH_DATA_PATH=data/semantic_meaning.json
```

The key stays on the server and is never sent to the browser.

## Project map

| Folder/file | Role |
| --- | --- |
| `app.py` | FastAPI application and endpoint wiring. |
| `chatbot.py` | Chat API, Groq client, request validation, and RAG orchestration. |
| `rag/` | Retrieval, prompt augmentation, safety, books, and article records. |
| `data/` | The 179-sign catalog and Gardiner mappings. |
| `models/` | YOLO checkpoint used by image detection. |
| `templates/` | Browser interface. |
| `tests/` | API and chatbot tests. |
| `docs/` | Project reports and UI documentation. |
| `notebooks/` | Training and speech experiments. |

## How chat works

1. The user sends a question from the browser.
2. `rag/retrieval.py` searches the local catalog, curated JSONL records, book
   metadata, and article records. Gardiner and Unicode matches receive priority.
3. If local evidence is weak and the question is a safe general-history question,
   the chatbot can add short Wikipedia summaries as web evidence.
4. `rag/augmentation.py` builds evidence blocks and citation tags.
5. `chatbot.py` sends the question and evidence to Groq.
6. The response includes citations, source links, similarity scores, and friendly
   follow-up suggestions.

The system does not silently scrape websites. Article records keep the source
URL and metadata; book records point to the local PDFs. Page-level extraction
should be added only when a page has been checked and cited.

## Tests

```powershell
python -m pytest -q
```

Health check:

```powershell
Invoke-WebRequest http://127.0.0.1:8000/health
```
