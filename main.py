#!/usr/bin/env python3
"""
LangChain + Amazon Bedrock + Hugging Face Embeddings (Chroma) — Student Starter (v2)

Models assumed available via Bedrock:
- Meta Llama 3 8B Instruct (default: meta.llama3-8b-instruct-v1:0)
- Mistral 7B Instruct (mistral.mistral-7b-instruct-v0:2)
- Mixtral 8x7B Instruct (mistral.mixtral-8x7b-instruct-v0:1)

Switch models by editing BEDROCK_MODEL_ID in .env.

Usage
-----
1. Create and activate a virtual environment:
   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate

2. Install the requirements:
   pip install -r requirements.txt

3. Create a .env file in the project root and fill in your AWS + Hugging Face values.

4. 4. Run the app:
   python main.py

"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_aws import ChatBedrock
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import uvicorn

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# Default to Llama 3 8B Instruct; students can change in .env
BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "meta.llama3-8b-instruct-v1:0")
AWS_REGION = os.getenv("AWS_DEFAULT_REGION") or os.getenv("AWS_REGION")

HF_EMBEDDING_MODEL = os.getenv("HF_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CHROMA_DIR = os.getenv("CHROMA_DIR", "chroma_db")

print("→ Initializing Hugging Face embeddings:", HF_EMBEDDING_MODEL)
embeddings = HuggingFaceEmbeddings(model_name=HF_EMBEDDING_MODEL, cache_folder=".hf_cache")

def load_local_documents() -> list[Document]:
    corpus_dir = Path(os.getenv("DOCS_DIR", "docs"))
    loaded: list[Document] = []
    if corpus_dir.exists() and corpus_dir.is_dir():
        for path in corpus_dir.rglob("*"):
            if path.suffix.lower() in {".txt", ".md"}:
                try:
                    loaded.extend(TextLoader(str(path), encoding="utf-8").load())
                except Exception:
                    pass
            elif path.suffix.lower() == ".pdf":
                try:
                    loaded.extend(PyPDFLoader(str(path)).load())
                except Exception:
                    pass
    if not loaded:
        # Fallback tiny corpus for first run
        loaded = [
            Document(page_content="LangChain is a framework for composing LLM apps using modular primitives like prompts, models, memory, and tools."),
            Document(page_content="Chroma is a lightweight vector database that stores embeddings so you can do semantic search and retrieval for RAG."),
            Document(page_content="Amazon Bedrock provides managed access to foundation models like Meta Llama and Mistral via a unified AWS API."),
            Document(page_content="Using embeddings, you can map text to vectors; similar texts have vectors close in space, enabling semantic search."),
        ]
    return loaded

def chunk_documents(docs: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    return splitter.split_documents(docs)

raw_docs = load_local_documents()
docs = chunk_documents(raw_docs)

vectordb = Chroma(
    collection_name="class_demo",
    embedding_function=embeddings,
    persist_directory=CHROMA_DIR,
)
_ = vectordb.add_documents(docs)
retriever = vectordb.as_retriever(search_kwargs={"k": 3})

app = FastAPI(title="LangChain + Bedrock + HF Embeddings Demo", version="1.0.0")

@app.post("/api/reindex")
def api_reindex():
    try:
        # Reload and re-chunk from docs folder
        new_raw = load_local_documents()
        new_docs = chunk_documents(new_raw)
        # Clear collection and add new docs
        try:
            vectordb.delete_collection()
        except Exception:
            # Fallback to deleting all vectors if wrapper lacks delete_collection
            try:
                vectordb._collection.delete(where={})  # type: ignore[attr-defined]
            except Exception:
                pass
        vectordb.add_documents(new_docs)
        return {"ok": True, "chunks": len(new_docs)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/clear")
def api_clear():
    try:
        try:
            vectordb.delete_collection()
        except Exception:
            try:
                vectordb._collection.delete(where={})  # type: ignore[attr-defined]
            except Exception:
                pass
        return {"ok": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class AskRequest(BaseModel):
    question: str
    k: int | None = 3

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <!doctype html>
    <html>
      <head>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <title>LangChain + Bedrock Demo</title>
        <style>
          body { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 40px; line-height: 1.5; }
          input, button, textarea { font-size: 16px; }
          .ans { white-space: pre-wrap; background:#f6f8fa; padding: 12px; border-radius: 8px; }
          .ctx { white-space: pre-wrap; background:#fffbe6; padding: 12px; border-radius: 8px; border: 1px solid #f0e6a6; }
          .row { margin: 12px 0; }
        </style>
      </head>
      <body>
        <h1>LangChain + Bedrock + HF Embeddings</h1>
        <p>Ask a question. If Bedrock isn’t configured, you’ll still see retrieved context.</p>
        <div class="row">
          <input id="q" type="text" placeholder="Your question..." style="width: 70%;" />
          <button onclick="ask()">Ask</button>
        </div>
        <div class="row"><strong>Answer</strong></div>
        <div id="answer" class="ans">(waiting...)</div>
        <div class="row"><strong>Retrieved Context</strong></div>
        <div id="context" class="ctx"></div>
        <div class="row"><button onclick="reindex()">Reindex from ./docs</button> <button onclick="clearDB()">Clear DB</button></div>
        <script>
          async function ask() {
            const q = document.getElementById('q').value;
            const res = await fetch('/api/ask', {
              method: 'POST',
              headers: {'Content-Type': 'application/json'},
              body: JSON.stringify({question: q})
            });
            const data = await res.json();
            document.getElementById('answer').textContent = data.answer || '(no LLM configured; see context below)';
            document.getElementById('context').textContent = data.context || '';
          }
          async function reindex(){
            await fetch('/api/reindex', {method:'POST'});
            alert('Reindex triggered.');
          }
          async function clearDB(){
            await fetch('/api/clear', {method:'POST'});
            alert('Collection cleared.');
          }
        </script>
      </body>
    </html>
    """

@app.post("/api/ask")
def api_ask(req: AskRequest):
    local_retriever = vectordb.as_retriever(search_kwargs={"k": req.k or 3})
    retrieved = local_retriever.invoke(req.question)
    context = format_docs(retrieved)

    if llm is None:
        return JSONResponse({"answer": None, "context": context})

    chain = prompt | llm
    try:
        resp = chain.invoke({"question": req.question, "context": context})
        return {"answer": resp.content, "context": context}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

SYSTEM_PROMPT = (
    "You are a concise teaching assistant. Answer based ONLY on the provided context. "
    "If the answer isn't in the context, say you don't know."
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "Question: {question}\n\nContext:\n{context}"),
    ]
)

def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])

def answer(question: str):
    retrieved = retriever.invoke(question)
    context = format_docs(retrieved)

    if llm is None:
        print("\n🔎 Retrieved context (no LLM configured):\n")
        for i, d in enumerate(retrieved, 1):
            print(f"[{i}] {d.page_content}")
        print("\n💡 Tip: Add AWS Bedrock creds to .env to enable LLM answers.\n")
        return

    chain = prompt | llm
    print("\n🧠 Bedrock LLM answer:\n")
    resp = chain.invoke({"question": question, "context": context})
    print(resp.content)

def make_bedrock_llm():
    if not AWS_REGION:
        print("⚠️  AWS region not set. Please set AWS_DEFAULT_REGION or AWS_REGION in your .env (e.g., us-east-1). Skipping Bedrock initialization.")
        return None
    try:
        llm = ChatBedrock(
            model=BEDROCK_MODEL_ID,
            region=AWS_REGION,
            beta_use_converse_api=True,
            # model_kwargs={"temperature": 0.2},
        )
        _ = llm.bind(stop=["\n"])
        return llm
    except Exception as e:
        print(f"⚠️  Could not initialize Bedrock LLM: {e}\nProceeding without an LLM.")
        return None

llm = make_bedrock_llm()

if __name__ == "__main__":
    # If a question is passed, run CLI mode; otherwise start a local FastAPI dev server.
    if len(sys.argv) >= 2:
        q = sys.argv[1]
        print(f"Q: {q}")
        answer(q)
    else:
        # Run with: python main.py  (then open http://127.0.0.1:8000/)
        uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)
