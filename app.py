import streamlit as st
from rag import add_documents, load_index, ask_question, reset_index

st.set_page_config(page_title="Knowlexa Demo", layout="centered")
st.title("📄 Knowlexa – AI Knowledge Assistant")

# -------- SESSION STATE INIT --------
if "index" not in st.session_state:
    st.session_state.index = None
if "metadata" not in st.session_state:
    st.session_state.metadata = []
if "chat" not in st.session_state:
    st.session_state.chat = []
if "indexed" not in st.session_state:
    st.session_state.indexed = False

# -------- LOAD INDEX ONCE --------
if not st.session_state.indexed:
    index, metadata = load_index()
    if index is not None:
        st.session_state.index = index
        st.session_state.metadata = metadata
        st.session_state.indexed = True

# -------- SIDEBAR --------
with st.sidebar:
    st.header("📂 Document Management")

    uploaded_files = st.file_uploader(
        "Upload PDF documents",
        type="pdf",
        accept_multiple_files=True
    )

    if uploaded_files and st.button("📌 Index Documents"):
        with st.spinner("Indexing documents..."):
            st.session_state.index, st.session_state.metadata = add_documents(
                uploaded_files,
                st.session_state.index,
                st.session_state.metadata
            )
            st.session_state.indexed = True
        st.success("Documents indexed successfully!")

    if st.button("🗑️ Reset Knowledge Base"):
        reset_index()
        st.session_state.index = None
        st.session_state.metadata = []
        st.session_state.chat = []
        st.session_state.indexed = False
        st.success("Knowledge base cleared.")

# -------- CHAT UI --------
st.subheader("💬 Ask Knowlexa")

if not st.session_state.indexed:
    st.info("Please upload and index documents before asking questions.")
else:
    question = st.chat_input("Ask a question from your documents")

    if question:
        answer, sources, _ = ask_question(
            question,
            st.session_state.index,
            st.session_state.metadata
        )

        st.session_state.chat.append({
            "question": question,
            "answer": answer,
            "sources": sources
        })

    for msg in st.session_state.chat:
        with st.chat_message("user"):
            st.write(msg["question"])

        with st.chat_message("assistant"):
            st.write(msg["answer"])
            st.caption(f"📌 Source: {', '.join(msg['sources'])}")
