"""Streamlit chat UI for session-scoped PDF Q&A.

Upload one or more PDFs, click Process, then ask questions. The vector store
is built in-memory for the current session only — nothing is persisted and
re-running the app starts fresh.
"""

import os
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

# .env lives at the project root (one level up from Frontend/).
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
EMBED_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"
TOP_K = 6

PROMPT_TEMPLATE = """You are a helpful assistant answering questions about the
user's uploaded documents.

Use ONLY the context below to answer the question. If the context does not
contain the answer, say "I don't know based on the provided documents." Do not
invent facts. Be concise. Cite sources inline as [source: <filename> p.<page>].

Context:
{context}

Question: {question}

Answer:"""


def build_chain(files):
    """Load PDFs, chunk, embed into an in-memory Chroma, return a QA chain."""
    docs = []
    with tempfile.TemporaryDirectory() as tmp:
        for f in files:
            path = os.path.join(tmp, f.name)
            with open(path, "wb") as out:
                out.write(f.getbuffer())
            docs.extend(PyPDFLoader(path).load())

    # Normalize `source` metadata to just the filename so citations stay clean
    # once the temp directory has been cleaned up.
    for d in docs:
        d.metadata["source"] = os.path.basename(d.metadata.get("source", ""))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(docs)

    # No persist_directory => ephemeral, session-scoped store.
    vs = Chroma.from_documents(
        chunks,
        embedding=OpenAIEmbeddings(model=EMBED_MODEL),
        collection_name="session",
    )

    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE, input_variables=["context", "question"]
    )
    document_prompt = PromptTemplate(
        template="[source: {source} p.{page}]\n{page_content}",
        input_variables=["source", "page", "page_content"],
    )
    return RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model=LLM_MODEL, temperature=0),
        chain_type="stuff",
        retriever=vs.as_retriever(search_kwargs={"k": TOP_K}),
        chain_type_kwargs={"prompt": prompt, "document_prompt": document_prompt},
        return_source_documents=True,
    )


st.set_page_config(page_title="Document Q&A", page_icon=None)
st.title("Document Q&A")

if "chain" not in st.session_state:
    st.session_state.chain = None
    st.session_state.messages = []
    st.session_state.doc_names = []

with st.sidebar:
    st.header("Documents")
    uploads = st.file_uploader(
        "Upload PDFs", type=["pdf"], accept_multiple_files=True
    )
    if st.button("Process", disabled=not uploads):
        if not os.getenv("OPENAI_API_KEY"):
            st.error("OPENAI_API_KEY is not set in the root .env file.")
        else:
            with st.spinner("Indexing documents..."):
                st.session_state.chain = build_chain(uploads)
                st.session_state.doc_names = [f.name for f in uploads]
                st.session_state.messages = []
            st.success(f"Indexed {len(uploads)} document(s).")

    if st.session_state.doc_names:
        st.caption("Active documents:")
        for n in st.session_state.doc_names:
            st.write(f"- {n}")

    if st.button("Clear chat", disabled=not st.session_state.messages):
        st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input(
    "Ask a question about your documents"
    if st.session_state.chain
    else "Upload PDFs and click Process to begin"
)

if question:
    if st.session_state.chain is None:
        st.warning("Please upload PDFs and click Process first.")
    else:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = st.session_state.chain.invoke({"query": question})
            answer = result["result"]
            sources = sorted({
                (d.metadata.get("source", ""), d.metadata.get("page"))
                for d in result["source_documents"]
            })
            if sources:
                answer += "\n\n**Sources:**\n" + "\n".join(
                    f"- {s} (page {p})" for s, p in sources
                )
            st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
