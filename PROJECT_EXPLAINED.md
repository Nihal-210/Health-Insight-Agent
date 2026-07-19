# Health Report Analyzer — Full Explanation & Interview Prep

This document assumes you know **zero** about RAG. By the end, you should be
able to explain this project to an interviewer without notes.

---

## PART 1 — What is RAG, actually? (No jargon first pass)

Imagine you ask an LLM: *"What does my hemoglobin level mean?"*

The LLM has never seen your report. It only knows general medical facts from
its training data. So it can explain what hemoglobin generally is, but it
has **no idea what your actual number is** unless you give it to them.

**Option A — "stuff everything in":** paste your entire report into every
single message, every time you ask a question. Works for short reports.
Breaks down when reports get long (multiple pages, many test panels) because:
- LLMs have a limited context window (a cap on how much text they can read at once)
- More text = more cost per API call (you pay per token)
- The model can get "distracted" by irrelevant parts of a long document

**Option B — RAG (Retrieval-Augmented Generation):** instead of sending the
whole document every time, you:
1. Store the document in a searchable form ahead of time
2. When a question comes in, **retrieve** only the small parts of the
   document that are actually relevant to that question
3. **Augment** the question with those relevant parts as context
4. Let the LLM **generate** an answer using only that context

That's it. RAG = Retrieve the relevant bits → Augment the prompt with them →
Generate the answer. It's "open-book exam" instead of "memorize the whole
textbook."

### How do you "search" a document for relevance?

Normal search (like Ctrl+F) matches exact words. That fails here — if you
ask "is my iron low?" but the report says "hemoglobin: 10.2 g/dL", there's
no shared keyword, yet it's exactly what you want.

So instead of matching words, we match **meaning**. This is done with
**embeddings**.

### What is an embedding?

An embedding model takes a piece of text and converts it into a **vector** —
a list of numbers (e.g. 384 numbers for the model we use). This list of
numbers is a mathematical representation of the text's *meaning*, positioned
in a high-dimensional space.

The key property: **texts with similar meaning end up with vectors that are
close together** in that space (measurable using distance, e.g. Euclidean
distance or cosine similarity). "Hemoglobin is low" and "iron levels are
reduced" would land close together, even though they don't share words.

### What is FAISS?

Once every chunk of your report has been converted into a vector, you have a
pile of number-lists. **FAISS** (Facebook AI Similarity Search) is a library
that stores these vectors and answers one question extremely fast: *"given
this new vector (the question), which stored vectors are closest to it?"*
That's the retrieval step.

### Putting it together — the full pipeline in this project

```
PDF uploaded
   │
   ▼
Extract raw text (pdfplumber)
   │
   ▼
Split text into small chunks (~500 characters each)
   │
   ▼
Convert each chunk into a vector (embedding model: all-MiniLM-L6-v2)
   │
   ▼
Store all vectors in a FAISS index  ──────────────┐
                                                    │  (this is the "retrieval"
User types a question                              │   half — happens once
   │                                                │   per report)
   ▼                                                │
Convert the question into a vector (same model)     │
   │                                                │
   ▼                                                │
Ask FAISS: "which chunks are closest to this?" ◄────┘
   │
   ▼
Take the top 3 closest chunks (not the whole report)
   │
   ▼
Build a prompt: "Here are relevant excerpts: [chunks]. Answer: [question]"
   │
   ▼
Send to the LLM (Groq / Llama 3.3 70B)
   │
   ▼
Answer, grounded only in the relevant part of the report
```

This is the entire concept of RAG. Everything else in the project is
plumbing around this idea (UI, storage, file handling).

---

## PART 2 — File by file, line by line

### `db.py` — the database

**Why it exists:** without a database, everything you do disappears the
moment you close the browser tab. The database is just permanent storage —
a single file on disk called `hia.db`.

**Why SQLite instead of Postgres/Supabase (like the original project):**
SQLite is built into Python (`import sqlite3`, no install needed), requires
no server, no account, no internet connection, and is literally one file.
Perfect for a single-user local project. Postgres/Supabase makes sense when
multiple users need concurrent access over a network — overkill here.

Walking through it:

```python
def get_connection():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn
```
- `sqlite3.connect(DB_FILE)` opens (or creates, if missing) the `hia.db` file.
- `conn.row_factory = sqlite3.Row` is a convenience setting: normally SQLite
  returns rows as plain tuples like `('report.pdf', 'text...')` where you'd
  have to remember `row[0]` is the filename. This setting lets you instead
  write `row["filename"]`, which is far less error-prone and easier to read.

```python
def init_db():
    conn = get_connection()
    conn.execute("""CREATE TABLE IF NOT EXISTS reports (...)""")
    conn.execute("""CREATE TABLE IF NOT EXISTS messages (...)""")
    conn.commit()
    conn.close()
```
- `CREATE TABLE IF NOT EXISTS` — this is idempotent, meaning you can call it
  every time the app starts without it erroring out or wiping existing data;
  it only creates the table the very first time.
- **`reports` table** — one row per uploaded report: an auto-incrementing
  `id`, the `filename`, the full `report_text`, the LLM's `analysis`, and a
  timestamp.
- **`messages` table** — one row per chat message. It has a `report_id`
  column that's a **foreign key** — meaning it points back to a specific row
  in `reports`. This is how the database knows "this chat message belongs to
  *this* report" — critical once you have multiple reports saved.
- `conn.commit()` — SQLite doesn't save changes to disk automatically after
  every command for performance reasons; `commit()` is you explicitly saying
  "yes, actually write this to the file now."

```python
def save_report(filename, report_text, analysis) -> int:
    conn = get_connection()
    cursor = conn.execute(
        "INSERT INTO reports (filename, report_text, analysis) VALUES (?, ?, ?)",
        (filename, report_text, analysis),
    )
    conn.commit()
    report_id = cursor.lastrowid
    conn.close()
    return report_id
```
- The `?` placeholders (instead of directly inserting the Python variables
  into the SQL string with f-strings) are a security practice called
  **parameterized queries** — it prevents **SQL injection**, where malicious
  text in a filename or report could otherwise be interpreted as SQL
  commands and corrupt or leak your database. *(Good thing to mention in an
  interview — shows security awareness even in a small project.)*
- `cursor.lastrowid` — after inserting, SQLite tells you the auto-generated
  `id` of the row you just created. We return that so the app can remember
  "this is report #7" for future chat messages.

`list_reports()`, `load_report()`, `load_messages()`, `save_message()` follow
the same pattern: open connection → run a `SELECT` or `INSERT` → return data
or close. Nothing more complex than that.

---

### `vector_search.py` — the RAG engine

```python
_model = None

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model
```
- Loading an embedding model from disk/downloading it is slow (a few
  seconds). If we reloaded it every time someone asked a question, the app
  would feel sluggish. This pattern — check if it's already loaded, load it
  once, reuse the same object — is called **lazy singleton loading**. The
  `global _model` keyword lets the function modify the module-level variable
  instead of creating a new local one.
- `all-MiniLM-L6-v2` is a small (~80MB), free, open-source embedding model
  from the `sentence-transformers` library. It converts any input text into
  a **384-dimensional vector** (a list of 384 numbers). It's not the most
  powerful embedding model available, but it's fast, runs locally (no API
  cost), and is the standard "default choice" for small RAG projects —
  exactly what the original `hia` repo used too.

```python
def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return [c.strip() for c in chunks if c.strip()]
```
- This slices the report text into overlapping windows of 500 characters.
- **Why overlap?** If a sentence gets cut exactly at character 500 (e.g.,
  "...hemoglobin level is | low, which may indicate..."), splitting it
  there loses meaning on both sides. Overlapping by 50 characters means each
  chunk shares a bit of text with its neighbor, reducing the chance an
  important sentence gets sliced in half with no context.
- `start += chunk_size - overlap` — this is what creates the overlap. If
  there were no overlap, you'd do `start += chunk_size`, giving you
  back-to-back, non-overlapping windows.
- The final list comprehension strips whitespace and drops any empty chunks.

```python
def build_index(chunks):
    model = get_model()
    vectors = model.encode(chunks)
    dimension = vectors.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(vectors)
    return index, chunks
```
- `model.encode(chunks)` — runs every chunk through the embedding model at
  once, returning a NumPy array of shape `(number_of_chunks, 384)` — one
  384-number vector per chunk.
- `faiss.IndexFlatL2(dimension)` — creates a FAISS index that does **exact**
  nearest-neighbor search using **L2 distance** (plain Euclidean distance —
  think of it as literal straight-line distance between two points, just in
  384-dimensional space instead of 2D). "Flat" means it doesn't use any
  approximation tricks — for a handful of chunks from one report, exact
  search is instant, so there's no need for the more complex approximate
  indexes FAISS also offers (like IVF or HNSW), which only start to matter
  at millions of vectors.
- `index.add(vectors)` — loads all the chunk vectors into the index so they
  can be searched.
- We return `chunks` alongside the `index` because FAISS only stores
  numbers — it has no idea what text a vector originally came from. We keep
  a parallel Python list so that when FAISS says "vector #2 is the closest
  match," we can look up `chunks[2]` to get the actual text back.

```python
def search(index, chunks, query, top_k=3):
    model = get_model()
    query_vector = model.encode([query])
    distances, indices = index.search(query_vector, top_k)
    return [chunks[i] for i in indices[0] if i < len(chunks)]
```
- Embed the user's question the exact same way the chunks were embedded —
  this is essential; you can only meaningfully compare vectors that came
  from the same embedding model.
- `index.search(query_vector, top_k)` — asks FAISS for the `top_k=3` closest
  chunks. It returns two arrays: `distances` (how far each match is — lower
  is more similar) and `indices` (which chunk positions those matches are).
- We only use `indices` here — mapping index positions back to the original
  chunk text.

---

### `app.py` — the Streamlit UI and orchestration

This file doesn't introduce new concepts — it wires together `db.py` and
`vector_search.py` behind a UI. Walking through the flow rather than every
line:

**On startup:**
```python
db.init_db()
```
Makes sure `hia.db` and its tables exist before anything else runs.

**Sidebar — past reports:**
```python
for r in past_reports:
    if st.sidebar.button(label, key=f"load_{r['id']}"):
        st.session_state.report_text, st.session_state.analysis = db.load_report(r["id"])
        st.session_state.chat_history = db.load_messages(r["id"])
        chunks = vector_search.chunk_text(st.session_state.report_text)
        st.session_state.index, st.session_state.chunks = vector_search.build_index(chunks)
```
- Note: we don't store the FAISS index or vectors in the database — only
  the raw report text. When you reload an old report, we **re-chunk and
  re-embed it from scratch**. This is a deliberate simplicity trade-off:
  persisting a FAISS index to disk and reloading it is extra complexity
  that isn't worth it for reports this small (re-embedding a report takes a
  fraction of a second). *(This is a great thing to flag in an interview as
  a conscious design decision, not an oversight — see Part 3.)*
- `st.session_state` is Streamlit's way of remembering values across user
  interactions during one browser session — normally, Streamlit reruns your
  entire script top-to-bottom on every click, so without `session_state`
  you'd lose everything (like the loaded report) on the very next click.

**Analyze button flow:**
1. Extract text from the uploaded PDF (`pdfplumber`)
2. Send that text to Groq with a system prompt instructing it to explain
   in plain language, flag abnormal values, and add a medical disclaimer
   — this call has **no retrieval step**, because the *first* analysis
   should cover the whole report, not just a relevant slice of it
3. Chunk + embed the report, build the FAISS index (this is where RAG
   setup happens)
4. Save the report + analysis to the database, store everything in
   `session_state`

**Chat flow (this is where RAG is actually used):**
1. User types a question into `st.chat_input`
2. `vector_search.search(...)` retrieves the top 3 most relevant chunks
3. `answer_followup()` builds a prompt containing only those chunks (not
   the full report) plus the conversation history, and sends it to Groq
4. The answer is displayed, then saved to both `session_state` (for
   immediate display) and the database (for persistence across restarts)

---

## PART 3 — Interview Q&A

**Q: What is RAG and why did you use it instead of just prompting the LLM
directly?**
> RAG stands for Retrieval-Augmented Generation. Instead of relying purely
> on the LLM's training knowledge, or stuffing an entire document into every
> prompt, you retrieve only the most relevant pieces of a document at query
> time and include those as context. I used it for the follow-up chat
> because it keeps each request smaller and cheaper, and keeps the model's
> answer grounded in the specific part of the report that's actually
> relevant to the question, rather than the whole document diluting its
> attention.

**Q: Walk me through your pipeline end to end.**
> PDF upload → text extraction with pdfplumber → for the initial analysis,
> the full text goes straight to the LLM. For chat, the text is chunked
> into ~500-character overlapping windows, embedded with a MiniLM sentence
> transformer into 384-dimensional vectors, and indexed in FAISS using flat
> L2 search. When a question comes in, it's embedded the same way, FAISS
> returns the top 3 nearest chunks, and those get passed to the LLM as
> context alongside the chat history.

**Q: Why FAISS specifically?**
> It's a lightweight, free, local library — no external vector database
> service or API cost. For a single report with a handful of chunks, a flat
> (exact, brute-force) index is more than fast enough; I didn't need
> approximate search structures like IVF or HNSW, which only pay off at
> much larger scale.

**Q: Why chunk size 500 with 50 overlap? How would you tune it?**
> It's a reasonable default for dense text like a lab report — small enough
> that each chunk is topically focused (so retrieval is precise), large
> enough to preserve context within a chunk. The overlap prevents a
> sentence from being split with no surrounding context on either side. In
> a real system I'd tune this empirically — measure retrieval quality on
> a labeled Q&A set, and consider chunking by semantic unit (e.g. per test
> panel/section) instead of a fixed character count, which is a known
> limitation of naive fixed-size chunking.

**Q: Why SQLite instead of the Postgres/Supabase setup from the original
project?**
> The original project needed multi-user auth and cloud persistence, so
> Postgres via Supabase made sense. This version is single-user and local,
> so SQLite — zero setup, one file, built into Python — is the right-sized
> tool. I'd swap it for Postgres the moment this needed to support multiple
> concurrent users or be deployed as a shared web service.

**Q: You don't persist the FAISS index — why?**
> Re-embedding a report on load takes well under a second since reports are
> short, so persisting and reloading a serialized FAISS index would add
> complexity without a meaningful performance benefit at this scale. If
> reports were much longer or this needed to scale to many users, I'd
> persist embeddings (e.g., in the SQLite DB as a serialized array, or
> in a proper vector database) rather than recomputing them every time.

**Q: What are the limitations of this system, and how would you improve
it?**
> - Fixed-size chunking can split meaningful sections awkwardly — semantic
>   chunking (splitting by section headers or sentence boundaries) would
>   improve retrieval quality.
> - No re-ranking step — I take FAISS's top-3 as final, but a re-ranker
>   model could improve precision on ambiguous questions.
> - No evaluation harness — in production I'd build a small labeled set of
>   question/expected-answer pairs to measure retrieval accuracy, not just
>   eyeball outputs.
> - No authentication or multi-user support — a deliberate simplification
>   here, but the natural next step if this needed real users.
> - Single embedding model, single LLM provider, no fallback — the original
>   `hia` project handled this with a model cascade through Groq; I removed
>   it here for simplicity, but it's the obvious production hardening step.

**Q: Why Groq instead of OpenAI/Anthropic?**
> Groq offers a generous free tier and very low latency inference for open
> models like Llama 3.3, which made it a practical choice for a personal
> project without ongoing API costs.

**Q: What's the difference between the "analysis" step and the "chat"
step in your app?**
> The analysis step is a single pass over the entire report — you want a
> comprehensive first read, so no retrieval is needed there. The chat step
> is where RAG applies: each follow-up question triggers a fresh retrieval
> against the report's chunks, so only relevant excerpts are sent, keeping
> answers focused and reducing token usage per request.
