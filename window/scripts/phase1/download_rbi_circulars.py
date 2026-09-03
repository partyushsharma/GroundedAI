import json
import re
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST_PATH = RAW_DIR / "manifest.jsonl"
MAX_FILES = 400

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122 Safari/537.36"
}

# RBI public index pages — update if RBI changes them
SEED_URLS = [
    "https://rbi.org.in/Scripts/BS_ViewMasterDirections.aspx",
    "https://rbi.org.in/Scripts/BS_ViewMasCirculdetails.aspx",
    # If you need a date-filtered list, add a URL like:
    # "https://rbi.org.in/Scripts/BS_ViewMasCirculdetails.aspx?frmdate=01/01/2023&todate=31/12/2025",
]

def get_soup(url):
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "html.parser")

def extract_date(a_tag):
    row = a_tag.find_parent("tr") or a_tag.find_parent("li") or a_tag.parent
    if row:
        text = row.get_text(" ", strip=True)
        m = re.search(r"\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b", text)
        if m:
            return m.group(0)
    return ""

def extract_pdf_links(soup, base_url):
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if not href:
            continue
        if ".pdf" in href.lower() or "rbidocs.rbi.org.in" in href.lower():
            full = urljoin(base_url, href)
            title = a.get_text(strip=True) or Path(urlparse(full).path).stem
            date = extract_date(a)
            links.append({
                "url": full,
                "title": title,
                "date": date,
                "source_page": base_url,
            })
    return links

def download_pdf(url, dest_dir):
    filename = Path(urlparse(url).path).name
    if not filename.lower().endswith(".pdf"):
        filename = f"{Path(urlparse(url).path).stem}.pdf"
    dest = dest_dir / filename

    if dest.exists():
        return dest

    try:
        resp = requests.get(url, headers=HEADERS, timeout=60, stream=True)
        resp.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        time.sleep(0.2)
        return dest
    except Exception as e:
        print(f"ERROR {url}: {e}")
        return None

def main():
    seen_urls = set()
    records = []

    for seed in SEED_URLS:
        print(f"Scraping {seed}")
        try:
            soup = get_soup(seed)
        except Exception as e:
            print(f"WARN failed {seed}: {e}")
            continue

        for item in extract_pdf_links(soup, seed):
            if item["url"] not in seen_urls:
                seen_urls.add(item["url"])
                records.append(item)

        time.sleep(0.5)

    records = records[:MAX_FILES]
    print(f"Found {len(records)} PDF links; downloading...")

    manifest_records = []
    for i, item in enumerate(records, 1):
        local_path = download_pdf(item["url"], RAW_DIR)
        if local_path:
            manifest_records.append({
                "url": item["url"],
                "title": item["title"],
                "date": item["date"],
                "file": str(local_path),
                "source_page": item["source_page"],
            })

        if i % 25 == 0:
            print(f"Downloaded {i}/{len(records)}")

    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        for rec in manifest_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Done. Downloaded {len(manifest_records)} PDFs to {RAW_DIR}")
    print(f"Manifest written to {MANIFEST_PATH}")

if __name__ == "__main__":
    main()