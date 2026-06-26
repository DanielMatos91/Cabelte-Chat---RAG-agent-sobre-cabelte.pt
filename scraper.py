"""
Crawl cabelte.pt e guarda todas as páginas em cabelte_pages.json
Uso: python scraper.py
"""
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import json
import time

BASE_URL = "https://www.cabelte.pt"
OUTPUT_FILE = "cabelte_pages.json"
MAX_PAGES = 100

SKIP_EXTENSIONS = {".pdf", ".jpg", ".jpeg", ".png", ".gif", ".svg", ".zip", ".doc", ".docx"}


def is_valid_url(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.netloc not in ("www.cabelte.pt", "cabelte.pt"):
        return False
    _, _, ext = parsed.path.rpartition(".")
    if "." in parsed.path.split("/")[-1] and f".{ext.lower()}" in SKIP_EXTENSIONS:
        return False
    return True


def clean_url(url: str) -> str:
    return url.split("#")[0].split("?")[0].rstrip("/")


def extract_text(soup: BeautifulSoup) -> str:
    for tag in soup(["nav", "footer", "script", "style", "header", "aside", "noscript"]):
        tag.decompose()
    return soup.get_text(separator="\n", strip=True)


def scrape_site():
    visited: set[str] = set()
    to_visit: list[str] = [BASE_URL]
    pages: list[dict] = []

    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 (compatible; CabelteChatBot/1.0; +https://cabelte.pt)"})

    print(f"A fazer scrape de {BASE_URL} (máx. {MAX_PAGES} páginas)...\n")

    while to_visit and len(visited) < MAX_PAGES:
        url = clean_url(to_visit.pop(0))
        if url in visited:
            continue

        try:
            resp = session.get(url, timeout=15)
            visited.add(url)

            if resp.status_code != 200:
                continue
            if "text/html" not in resp.headers.get("content-type", ""):
                continue

            soup = BeautifulSoup(resp.text, "html.parser")
            title = soup.title.string.strip() if soup.title and soup.title.string else url
            text = extract_text(soup)

            if len(text) > 150:
                pages.append({"url": url, "title": title, "text": text})
                print(f"  [{len(pages):3d}] {title[:70]}")

            for a in soup.find_all("a", href=True):
                candidate = clean_url(urljoin(url, a["href"]))
                if candidate not in visited and is_valid_url(candidate):
                    to_visit.append(candidate)

            time.sleep(0.4)

        except Exception as e:
            print(f"  [ERRO] {url}: {e}")

    print(f"\nTotal: {len(pages)} páginas recolhidas.")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(pages, f, ensure_ascii=False, indent=2)

    print(f"Guardado em {OUTPUT_FILE}")
    return pages


if __name__ == "__main__":
    scrape_site()
