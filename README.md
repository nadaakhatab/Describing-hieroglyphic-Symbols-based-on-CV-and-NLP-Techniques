# Hieroglyph Assistant

This project is a local FastAPI application for learning about Ancient Egyptian hieroglyphs. It has three main features:

- A chatbot that explains Gardiner codes and Unicode hieroglyph symbols.
- An image-detection endpoint that uses the saved YOLO model.
- A speech endpoint that reads a local description and tries to translate it into Arabic.

The chatbot uses the descriptions in `data/Semantic meaning.json`. It can answer in English or Arabic. The Groq API key stays on the server in `.env`; it is never sent to the browser.

## Project structure

| Path | Purpose |
| --- | --- |
| `app.py` | Main FastAPI application. It connects chat, detection, and speech. |
| `chatbot/` | Compact chatbot package: settings, request schema, local search, Groq service, and API route/UI. |
| `data/Semantic meaning.json` | Local descriptions for 179 hieroglyph records. |
| `models/best.pt` | Saved YOLO detection model. |
| `notebooks/` | Training and text-to-speech experiment notebooks. They are not needed to start the chatbot. |
| `tests/` | Automated tests for chat and local data lookup. |
| `requirements-chatbot.txt` | Packages needed to run the chatbot and FastAPI server. |
| `.env.example` | Safe example settings file. Copy it to `.env` and add your own Groq key. |

## Start the chatbot

Open PowerShell in the project folder and run:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-chatbot.txt
.\.venv\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Then open [http://127.0.0.1:8000](http://127.0.0.1:8000).

## Add your Groq key

The chatbot can start without a key, but it cannot generate an AI answer until you add one.

1. Copy `.env.example` to a new file named `.env`.
2. Open `.env` locally.
3. Replace `your_groq_api_key_here` with your key from Groq.
4. Restart the server.

Example:

```env
GROQ_API_KEY=your_real_key_here
GROQ_MODEL=openai/gpt-oss-20b
GLYPH_DATA_PATH=data/Semantic meaning.json
```

Do not share the key in chat and do not commit `.env`. Git ignores it.

## Test the project

Run the automated tests:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

You can also check the server:

```powershell
Invoke-WebRequest http://127.0.0.1:8000/health
```

Expected result: JSON with `"status": "API is running"` and `"story_records": 179`.

## Try the chatbot in the browser

Use these tests after the server is running and your Groq key is configured:

| Glyph code | Question | Expected result |
| --- | --- | --- |
| `A1` | `What does A1 mean?` | An explanation based on the local A1 story. |
| `S34` | `Explain this symbol in Arabic.` | An Arabic explanation of the ankh and life. |
| `𓀀` | `Tell me more about this sign.` | The same local record as A1. |
| `ZZ999` | `What does this code mean?` | A clear message that no local description was found. |

The chatbot does not read an image by itself. Enter a known Gardiner code or Unicode symbol in the **Glyph codes** box. The image detector currently returns an annotated image, not a list of codes.

## Important notes

- The detector and speech features load their heavy packages only when you call their endpoints. This keeps the chatbot fast to start.
- Detection needs extra computer-vision packages and may need model downloads. Speech needs Kokoro, SoundFile, Google Translate access, and its model files.
- The stored descriptions are project data. They should be checked by an Egyptology expert before being used as final academic information.
- The live Groq response depends on network access, your API key, and your free-plan limit.
