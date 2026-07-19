"""
Simple Health Report Analyzer
------------------------------
Upload a medical/blood report PDF -> get a plain-language explanation ->
ask follow-up questions in a chat box, answered using vector search (RAG)
over the report. Past reports and chats are saved in a local SQLite database.

No login. Just open the app and go.
"""

import streamlit as st
import pdfplumber
from groq import Groq

import db
import vector_search

# ---------------------------------------------------------------------
# 1. PAGE SETUP + DATABASE INIT
# ---------------------------------------------------------------------
st.set_page_config(page_title="Health Report Analyzer", page_icon="🩺")
st.title("🩺 Health Report Analyzer")
st.caption("Upload a medical report PDF and get a plain-language explanation.")

db.init_db()  # creates hia.db and its tables if they don't exist yet

MODEL = "llama-3.3-70b-versatile"

# ---------------------------------------------------------------------
# 2. API KEY
# ---------------------------------------------------------------------
st.sidebar.header("Setup")
api_key = st.sidebar.text_input("Groq API Key", type="password")
st.sidebar.caption("Get a free key at console.groq.com")

# ---------------------------------------------------------------------
# 3. SIDEBAR — past reports, loaded from the database
# ---------------------------------------------------------------------
st.sidebar.divider()
st.sidebar.header("Past Reports")
past_reports = db.list_reports()

for r in past_reports:
    label = f"{r['filename']} ({r['created_at'][:16]})"
    if st.sidebar.button(label, key=f"load_{r['id']}"):
        st.session_state.report_id = r["id"]
        st.session_state.report_text, st.session_state.analysis = db.load_report(r["id"])
        st.session_state.chat_history = db.load_messages(r["id"])
        # Rebuild the vector index for this report's chunks (we don't store
        # the vectors themselves in the DB — just re-embed on load; it's fast)
        chunks = vector_search.chunk_text(st.session_state.report_text)
        st.session_state.index, st.session_state.chunks = vector_search.build_index(chunks)
        st.rerun()

# ---------------------------------------------------------------------
# 4. SESSION STATE — values that persist while the app tab is open
# ---------------------------------------------------------------------
for key in ["report_id", "report_text", "analysis", "chat_history", "index", "chunks"]:
    if key not in st.session_state:
        st.session_state[key] = None
if st.session_state.chat_history is None:
    st.session_state.chat_history = []


# ---------------------------------------------------------------------
# 5. HELPER FUNCTIONS
# ---------------------------------------------------------------------
def extract_text_from_pdf(uploaded_file) -> str:
    """Turn the uploaded PDF into plain text."""
    text = ""
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text


def call_groq(client, messages):
    """Send a list of chat messages to Groq and return the reply text."""
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.3,
    )
    return response.choices[0].message.content


def analyze_report(client, report_text: str) -> str:
    """Ask the LLM to explain the report in plain language."""
    system_prompt = (
        "You are a friendly medical assistant. Explain the following blood/health "
        "report in simple, plain language for someone with no medical background. "
        "Structure your answer with: 1) A short overall summary, 2) Any values that "
        "are outside the normal range and what that might mean, 3) General, "
        "non-prescriptive lifestyle suggestions. Always remind the reader this is "
        "not a substitute for professional medical advice."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": report_text},
    ]
    return call_groq(client, messages)


def answer_followup(client, relevant_chunks: list, chat_history: list, question: str) -> str:
    """Answer a follow-up question, grounded only in the most relevant report chunks."""
    context = "\n---\n".join(relevant_chunks)
    system_prompt = (
        "You are a friendly medical assistant helping someone understand their "
        "health report. Base your answer only on the report excerpts below. If the "
        "answer isn't in them, say so honestly. Keep answers short and in plain "
        "language.\n\n"
        f"--- RELEVANT REPORT EXCERPTS ---\n{context}\n--- END EXCERPTS ---"
    )
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(chat_history)
    messages.append({"role": "user", "content": question})
    return call_groq(client, messages)


# ---------------------------------------------------------------------
# 6. MAIN APP FLOW — new report
# ---------------------------------------------------------------------
uploaded_file = st.file_uploader("Upload your report (PDF)", type=["pdf"])

if uploaded_file and st.button("Analyze Report"):
    if not api_key:
        st.error("Please paste your Groq API key in the sidebar first.")
    else:
        client = Groq(api_key=api_key)
        with st.spinner("Reading your report..."):
            report_text = extract_text_from_pdf(uploaded_file)

        if not report_text.strip():
            st.error("Couldn't find any readable text in that PDF. Try a different file.")
        else:
            with st.spinner("Analyzing..."):
                analysis = analyze_report(client, report_text)

            with st.spinner("Indexing report for chat..."):
                chunks = vector_search.chunk_text(report_text)
                index, chunks = vector_search.build_index(chunks)

            # Save everything to the database
            report_id = db.save_report(uploaded_file.name, report_text, analysis)

            # Update session state
            st.session_state.report_id = report_id
            st.session_state.report_text = report_text
            st.session_state.analysis = analysis
            st.session_state.index = index
            st.session_state.chunks = chunks
            st.session_state.chat_history = []
            st.rerun()

# ---------------------------------------------------------------------
# 7. SHOW ANALYSIS + CHAT (for whichever report is currently loaded)
# ---------------------------------------------------------------------
if st.session_state.analysis:
    st.subheader("📋 Plain-Language Analysis")
    st.write(st.session_state.analysis)

    st.divider()
    st.subheader("💬 Ask a follow-up question")

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    question = st.chat_input("e.g. What does my cholesterol number mean?")
    if question:
        if not api_key:
            st.error("Please paste your Groq API key in the sidebar first.")
        else:
            client = Groq(api_key=api_key)

            with st.chat_message("user"):
                st.write(question)

            # Vector search: find the report chunks most relevant to this question
            relevant_chunks = vector_search.search(
                st.session_state.index, st.session_state.chunks, question
            )

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    answer = answer_followup(
                        client, relevant_chunks, st.session_state.chat_history, question
                    )
                st.write(answer)

            # Update in-memory history (for display) and save to the database (for persistence)
            st.session_state.chat_history.append({"role": "user", "content": question})
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
            db.save_message(st.session_state.report_id, "user", question)
            db.save_message(st.session_state.report_id, "assistant", answer)
