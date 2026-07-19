# Simple Health Report Analyzer

Upload a medical report PDF, get a plain-language explanation, and ask
follow-up questions in a chat box. No login required.

## How it works

You upload a PDF → we extract its text → an LLM (via Groq) rewrites it in
plain language. For the chat, we split the report into small chunks,
turn each chunk into a "vector" (a list of numbers representing its
meaning) using a local embedding model, and store those vectors in FAISS.
When you ask a question, we turn the question into a vector too and ask
FAISS which report chunks are the closest match — those chunks (not the
whole report) get sent to the LLM to answer you. Everything (reports +
chats) is saved in a local SQLite file, `hia.db`, so it's still there
next time you open the app.

## Files

| File | What it does |
|---|---|
| `app.py` | The Streamlit UI and main app flow |
| `db.py` | SQLite database — saves/loads reports and chat messages |
| `vector_search.py` | Chunking, embedding, and FAISS search for the chat |
| `requirements.txt` | Python packages needed |

## Setup

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Get a free Groq API key at [console.groq.com](https://console.groq.com)

3. Run the app:
   ```
   streamlit run app.py
   ```

4. Paste your Groq API key into the sidebar, upload a PDF, click
   **Analyze Report**, and start asking questions.

## Notes

- The first time you run it, the embedding model (~80MB) will download
  automatically — that's normal, it only happens once.
- `hia.db` is created automatically in the same folder the first time you
  run the app. Delete it any time to wipe all saved history.

