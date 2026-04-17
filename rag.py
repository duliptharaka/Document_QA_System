"""Single-script RAG pipeline over PDF research papers.

Flow: Load PDFs -> Chunk -> Embed & store in Chroma -> Retrieve -> LLM answer
      -> Evaluate against eval_set.json (retrieval / faithfulness / correctness).
"""

import json
import os

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

load_dotenv()

DOCS_DIR = "Documents"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
EMBED_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"
PERSIST_DIR = "chroma_db"
COLLECTION = "papers"
EVAL_FILE = "eval_set.json"
# k=6 because top-4 sometimes pulled title/references pages and starved the LLM
# of body content, producing "I don't know" on otherwise-answerable questions.
TOP_K = 6

PROMPT_TEMPLATE = """You are a helpful research assistant answering questions about
electric power transmission network research papers.

Use ONLY the context below to answer the question. If the context does not contain
the answer, say "I don't know based on the provided documents." Do not invent facts.
Be concise and technical. Cite sources inline as [source: <filename> p.<page>].

Context:
{context}

Question: {question}

Answer:"""


def load_documents():
    # PyPDFDirectoryLoader yields ONE Document per page (not per file).
    loader = PyPDFDirectoryLoader(DOCS_DIR)
    return loader.load()


def chunk_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_documents(docs)


def build_vector_store(chunks):
    embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
    vs = Chroma(
        collection_name=COLLECTION,
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR,
    )
    # Only embed on first run. Delete chroma_db/ to force a rebuild after
    # changing chunking or the embedding model.
    if vs._collection.count() == 0:
        vs.add_documents(chunks)
    return vs


def test_retrieval(vs, queries, k=3):
    """Pure retrieval sanity check (no LLM) — validates embedding quality."""
    for q in queries:
        print("\n" + "=" * 80)
        print(f"QUERY: {q}")
        print("=" * 80)
        hits = vs.similarity_search(q, k=k)
        for i, h in enumerate(hits, 1):
            src = os.path.basename(h.metadata.get("source", ""))
            page = h.metadata.get("page")
            preview = h.page_content.strip().replace("\n", " ")[:300]
            print(f"\n[{i}] {src} (page {page})")
            print(f"    {preview}...")


def build_qa_chain(vs):
    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"],
    )
    # document_prompt wraps each retrieved chunk with its source/page BEFORE
    # it reaches the LLM, so inline citations resolve to real filenames
    # instead of the literal "<filename>" placeholder.
    document_prompt = PromptTemplate(
        template="[source: {source} p.{page}]\n{page_content}",
        input_variables=["source", "page", "page_content"],
    )
    llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vs.as_retriever(search_kwargs={"k": TOP_K}),
        chain_type_kwargs={"prompt": prompt, "document_prompt": document_prompt},
        return_source_documents=True,
    )


def run_qa(chain, queries):
    for q in queries:
        print("\n" + "=" * 80)
        print(f"QUESTION: {q}")
        print("=" * 80)
        result = chain.invoke({"query": q})
        print(f"\nANSWER:\n{result['result']}")
        print("\nSOURCES:")
        seen = set()
        for d in result["source_documents"]:
            key = (os.path.basename(d.metadata.get("source", "")),
                   d.metadata.get("page"))
            if key not in seen:
                seen.add(key)
                print(f"  - {key[0]} (page {key[1]})")


def _grade_item(item, result):
    """Auto-grade one eval item. Manual review is still recommended."""
    answer = result["result"]
    answer_lc = answer.lower()
    retrieved_files = {
        os.path.basename(d.metadata.get("source", ""))
        for d in result["source_documents"]
    }
    refused = "i don't know" in answer_lc

    if item["refusal_expected"]:
        # For refusal items, retrieval grading is N/A — we expect the model to
        # recognize the topic is absent regardless of what came back.
        retrieval_ok = True
        faithfulness_ok = refused
        correctness_ok = refused
    else:
        retrieval_ok = any(s in retrieved_files for s in item["expected_sources"])
        # Faithful = didn't refuse AND cited at least one retrieved file inline.
        cited_retrieved = any(s in answer for s in retrieved_files)
        faithfulness_ok = (not refused) and cited_retrieved
        correctness_ok = all(kw in answer_lc for kw in item["expected_keywords"])

    return {
        "retrieval": retrieval_ok,
        "faithfulness": faithfulness_ok,
        "correctness": correctness_ok,
        "answer": answer,
        "retrieved_files": sorted(retrieved_files),
    }


def evaluate(chain, eval_path=EVAL_FILE):
    with open(eval_path, "r", encoding="utf-8") as f:
        items = json.load(f)

    tick = lambda b: "PASS" if b else "FAIL"
    rows = []
    for item in items:
        result = chain.invoke({"query": item["question"]})
        g = _grade_item(item, result)
        rows.append((item, g))
        print("\n" + "-" * 80)
        print(f"Q{item['id']}: {item['question']}")
        print(f"Expected: {item['expected_answer']}")
        print(f"Got:      {g['answer']}")
        print(f"Retrieved files: {g['retrieved_files']}")
        print(f"  Retrieval:    {tick(g['retrieval'])}")
        print(f"  Faithfulness: {tick(g['faithfulness'])}")
        print(f"  Correctness:  {tick(g['correctness'])}")

    n = len(rows)
    r = sum(1 for _, g in rows if g["retrieval"])
    f_ = sum(1 for _, g in rows if g["faithfulness"])
    c = sum(1 for _, g in rows if g["correctness"])
    print("\n" + "=" * 80)
    print(f"SCORECARD  (n={n})")
    print("=" * 80)
    print(f"  Retrieval:    {r}/{n}")
    print(f"  Faithfulness: {f_}/{n}")
    print(f"  Correctness:  {c}/{n}")
    print("\n(Auto-graded — please double-check answers manually against the PDFs.)")


if __name__ == "__main__":
    # Step 1: Load
    docs = load_documents()
    unique_files = {d.metadata["source"] for d in docs}
    print(f"Number of PDF files: {len(unique_files)}")
    print(f"Number of pages (LangChain Documents): {len(docs)}")

    # Step 2: Chunk
    chunks = chunk_documents(docs)
    sizes = [len(c.page_content) for c in chunks]
    print(f"\nTotal chunks created: {len(chunks)}")
    print(f"Smallest chunk: {min(sizes)} chars")
    print(f"Largest chunk:  {max(sizes)} chars")

    # Step 3: Embed + Store
    vs = build_vector_store(chunks)
    print(f"\nEmbedding model: {EMBED_MODEL}")
    print(f"Vector store: ChromaDB (persisted at '{PERSIST_DIR}')")
    print(f"Vectors stored: {vs._collection.count()}")

    # Step 4: Test retrieval (before wiring up the LLM)
    test_queries = [
        "How is electric power transmission network topology processed efficiently?",
        "How are graph models of power networks constructed from synchrophasor PMU data?",
        "How does an LVQ neural network identify power system network branch events online?",
    ]
    test_retrieval(vs, test_queries, k=3)

    # Step 5: Full RAG chain (retrieval + LLM answer)
    print("\n\n" + "#" * 80)
    print(f"# RAG Q&A with {LLM_MODEL}")
    print("#" * 80)
    qa_chain = build_qa_chain(vs)
    run_qa(qa_chain, test_queries)

    # Step 6: Evaluate against a mini eval set
    print("\n\n" + "#" * 80)
    print(f"# EVALUATION ({EVAL_FILE})")
    print("#" * 80)
    evaluate(qa_chain)
