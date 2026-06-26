"""
Lê cabelte_pages.json, divide em chunks, cria embeddings via Google Gemini
e guarda no ChromaDB (pasta chroma_db/).
Uso: python ingest.py
Requer: GEMINI_API_KEY em .streamlit/secrets.toml ou variável de ambiente
"""
import json
import sys
import os
import time
import chromadb
from google import genai
from google.genai import types

PAGES_FILE = "cabelte_pages.json"
CHROMA_DIR = "chroma_db"
EMBED_MODEL = "gemini-embedding-001"
CHUNK_SIZE = 600
CHUNK_OVERLAP = 100


def get_api_key() -> str:
    key = os.environ.get("GEMINI_API_KEY", "")
    if not key:
        try:
            import tomllib
            with open(".streamlit/secrets.toml", "rb") as f:
                secrets = tomllib.load(f)
            key = secrets.get("GEMINI_API_KEY", "")
        except Exception:
            pass
    if not key or "cole-aqui" in key:
        print("ERRO: GEMINI_API_KEY não configurada.")
        print("Edita .streamlit/secrets.toml e substitui o placeholder pela tua chave.")
        sys.exit(1)
    return key


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
    api_key = get_api_key()
    client = genai.Client(api_key=api_key)

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
    print(f"A gerar embeddings com {EMBED_MODEL}...")
    print(f"(limite gratuito: 100 pedidos/min — ritmo controlado automaticamente)\n")

    db = chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        db.delete_collection("cabelte")
        print("Colecção anterior removida.")
    except Exception:
        pass
    collection = db.create_collection("cabelte")

    for idx, (chunk, meta, cid) in enumerate(zip(all_chunks, all_metas, all_ids), 1):
        for attempt in range(3):
            try:
                result = client.models.embed_content(
                    model=EMBED_MODEL,
                    contents=chunk,
                    config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
                )
                emb = result.embeddings[0].values
                collection.add(documents=[chunk], embeddings=[emb], metadatas=[meta], ids=[cid])
                break
            except Exception as e:
                if attempt < 2:
                    time.sleep(5)
                else:
                    print(f"  [ERRO] chunk {cid}: {e}")

        time.sleep(0.65)  # ~92 pedidos/min, abaixo do limite de 100 RPM

        if idx % 20 == 0 or idx == len(all_chunks):
            print(f"  {idx}/{len(all_chunks)} chunks processados...")

    print(f"\nIngestão completa! {len(all_chunks)} chunks guardados em {CHROMA_DIR}/")


if __name__ == "__main__":
    ingest()
