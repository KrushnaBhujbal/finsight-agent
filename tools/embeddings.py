import os
import logging
import chromadb
from chromadb.utils import embedding_functions

log = logging.getLogger("finsight.rag")

CHROMA_DIR = "chroma_db"
EMBED_MODEL = "all-MiniLM-L6-v2"

def get_chroma_client():
    log.info(f"Connecting to ChromaDB at ./{CHROMA_DIR}")
    return chromadb.PersistentClient(path=CHROMA_DIR)

def get_embedding_function():
    log.info(f"Loading embedding model: {EMBED_MODEL}")
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL
    )

def get_or_create_collection(ticker: str):
    client = get_chroma_client()
    ef = get_embedding_function()
    name = f"filing_{ticker.lower()}"
    collection = client.get_or_create_collection(
        name=name,
        embedding_function=ef,
        metadata={"ticker": ticker}
    )
    log.info(f"Collection '{name}' ready — {collection.count()} chunks")
    return collection

def chunk_text(text: str, chunk_size: int = 512, overlap: int = 50) -> list[str]:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        if len(chunk.strip()) > 50:
            chunks.append(chunk)
        start += chunk_size - overlap
    log.info(f"Split text into {len(chunks)} chunks ({chunk_size} words, {overlap} overlap)")
    return chunks

def embed_filing(ticker: str, text: str, filing_date: str) -> int:
    collection = get_or_create_collection(ticker)

    if collection.count() > 0:
        log.info(f"Collection already has {collection.count()} chunks — skipping embed")
        return collection.count()

    chunks = chunk_text(text)
    ids = [f"{ticker}_{i}" for i in range(len(chunks))]
    metadatas = [{"ticker": ticker, "chunk_idx": i, "filing_date": filing_date} for i in range(len(chunks))]

    log.info(f"Embedding {len(chunks)} chunks into ChromaDB...")
    collection.add(documents=chunks, ids=ids, metadatas=metadatas)
    log.info(f"Embedded {len(chunks)} chunks for {ticker}")
    return len(chunks)

def query_filing(ticker: str, question: str, n_results: int = 3) -> list[dict]:
    collection = get_or_create_collection(ticker)

    if collection.count() == 0:
        log.warning(f"No chunks found for {ticker} — run embed_filing first")
        return []

    log.info(f"Querying '{question[:50]}...' against {collection.count()} chunks")
    results = collection.query(
        query_texts=[question],
        n_results=min(n_results, collection.count())
    )

    hits = []
    for i, doc in enumerate(results["documents"][0]):
        hits.append({
            "chunk": doc,
            "chunk_idx": results["metadatas"][0][i]["chunk_idx"],
            "distance": results["distances"][0][i]
        })

    log.info(f"Retrieved {len(hits)} relevant chunks")
    return hits