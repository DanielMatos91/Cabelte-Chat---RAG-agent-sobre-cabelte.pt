"""
Lê cabelte_pages.json, divide em chunks, cria embeddings locais (sentence-transformers)
e guarda no ChromaDB (pasta chroma_db/).
Uso: python ingest.py
Não requer chave API — o modelo de embeddings corre localmente.
"""
import json
import sys
import chromadb
from sentence_transformers import SentenceTransformer

PAGES_FILE = "cabelte_pages.json"
CHROMA_DIR = "chroma_db"
EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
CHUNK_SIZE = 600
CHUNK_OVERLAP = 100


def split_text(text: str) -> list[str]:
    paragraphs = [p.strip() for p in text.split("\n") if len(p.strip()) > 30]
    chunks, current = [], ""
    for para in paragraphs:
        if len(current) + len(para) < CHUNK_SIZE:
            current += (" " if current else "") + para
        else:
            if current:
                chunks.append(current)
            while len(para) > CHUNK_SIZE:
                chunks.append(para[:CHUNK_SIZE])
                para = para[CHUNK_SIZE - CHUNK_OVERLAP:]
            current = para
    if current:
        chunks.append(current)
    return chunks


def ingest():
    print(f"A carregar modelo de embeddings local ({EMBED_MODEL})...")
    model = SentenceTransformer(EMBED_MODEL)

    try:
        with open(PAGES_FILE, "r", encoding="utf-8") as f:
            pages = json.load(f)
    except FileNotFoundError:
        print(f"Ficheiro {PAGES_FILE} não encontrado. Corre primeiro: python scraper.py")
        sys.exit(1)

    print(f"A processar {len(pages)} páginas...")

    all_chunks, all_metas, all_ids = [], [], []
    for i, page in enumerate(pages):
        chunks = split_text(page["text"])
        for j, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_metas.append({"url": page["url"], "title": page["title"]})
            all_ids.append(f"p{i}_c{j}")

    print(f"Total de chunks: {len(all_chunks)}")
    print("A gerar embeddings (local, sem limites de pedidos)...")

    db = chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        db.delete_collection("cabelte")
        print("Colecção anterior removida.")
    except Exception:
        pass
    collection = db.create_collection("cabelte")

    embeddings = model.encode(all_chunks, show_progress_bar=True, batch_size=32).tolist()
    collection.add(documents=all_chunks, embeddings=embeddings, metadatas=all_metas, ids=all_ids)

    print(f"\nIngestão completa! {len(all_chunks)} chunks guardados em {CHROMA_DIR}/")


if __name__ == "__main__":
    ingest()
