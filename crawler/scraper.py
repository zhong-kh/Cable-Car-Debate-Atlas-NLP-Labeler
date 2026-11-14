import argparse
import hashlib
import json
import os
import re
from email.utils import parsedate_to_datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
import requests
import feedparser
import trafilatura
from urllib.parse import quote_plus
from bs4 import BeautifulSoup

DEFAULT_OUTPUT = Path(__file__).resolve().parent.parent / "data" / "incoming" / "heilbronn_cable_car_data.json"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
}

# ---------------------------------------------------------------------------
# Relevance filter (German + English)
# ---------------------------------------------------------------------------
CABLE_CAR_DE = re.compile(
    r"seilbahn|kabinenbahn|gondelbahn|luftseilbahn|seilschwebebahn|\bgondel",
    re.IGNORECASE,
)
CABLE_CAR_EN = re.compile(
    r"\bcable[\s-]?cars?\b|\bcableways?\b|\bgondolas?\b|\bgondola[\s-]lifts?\b"
    r"|\baerial[\s-](?:tram|tramway|lift|ropeway)s?\b|\bropeways?\b",
    re.IGNORECASE,
)
LOCATION = re.compile(
    r"heilbronn|\bwartberg\b|\bbuga\b|bundesgartenschau|federal horticultural show"
    r"|garden show|neckarbogen",
    re.IGNORECASE,
)

GERMAN_STOPWORDS = {"der", "die", "das", "und", "ist", "nicht", "mit", "für", "auf", "ein", "eine", "wird", "von", "den", "dem", "zu", "auch"}
ENGLISH_STOPWORDS = {"the", "and", "is", "not", "with", "for", "on", "a", "an", "will", "of", "to", "in", "that", "this", "are", "was"}


def is_relevant(*texts):
    combined = " ".join(t for t in texts if t)
    has_cable_car = bool(CABLE_CAR_DE.search(combined) or CABLE_CAR_EN.search(combined))
    return has_cable_car and bool(LOCATION.search(combined))


def detect_language(text):
    words = re.findall(r"[a-zäöüß]+", text.lower())[:400]
    de = sum(w in GERMAN_STOPWORDS for w in words)
    en = sum(w in ENGLISH_STOPWORDS for w in words)
    if de == en == 0:
        return "unknown"
    return "de" if de > en else "en"


def make_record(url, timestamp, media_type, content, source):
    timestamp = timestamp or ""
    try:
        published_at = parsedate_to_datetime(timestamp).isoformat() if timestamp else ""
    except (TypeError, ValueError, OverflowError):
        published_at = timestamp
    language = detect_language(content)
    title = content.partition("\n\n")[0].strip()
    identity = url.strip() or f"{source}\n{timestamp}\n{content}"
    mention_id = "crawler:" + hashlib.sha256(identity.encode("utf-8")).hexdigest()
    return {
        "mention_id": mention_id,
        "title": title,
        "published_at": published_at,
        "media_type_name": media_type,
        "source_name": source,
        "language_code": language,
        "url": url,
        "timestamp": timestamp,
        "media_type": media_type,
        "source": source,
        "language": language,
        "content": content,
    }


def clean_html(raw):
    return BeautifulSoup(raw or "", "html.parser").get_text(" ", strip=True)


def fetch_article_text(url):
    """Download a page and extract the main article text (open source, no key)."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        text = trafilatura.extract(resp.text, include_comments=False)
        if text:
            return text
        soup = BeautifulSoup(resp.content, "html.parser")
        return " ".join(p.get_text(strip=True) for p in soup.select("article p"))
    except Exception as e:
        print(f"Failed to fetch article {url}: {e}")
        return ""


# ---------------------------------------------------------------------------
# Sources (none of these need an API key)
# ---------------------------------------------------------------------------
def crawl_swr():
    records = []
    feed = feedparser.parse("https://www.swr.de/~rss/swraktuell/swraktuell-bw-100.xml")

    for entry in feed.entries:
        summary = clean_html(entry.get("summary", ""))
        hint = (entry.link + entry.title + summary).lower()
        if not (is_relevant(entry.title, summary) or "heilbronn" in hint):
            continue

        body = fetch_article_text(entry.link)
        if not is_relevant(entry.title, summary, body):
            continue
        records.append(make_record(
            entry.link, entry.get("published", ""), "news_article",
            f"{entry.title}\n\n{body}", "SWR Aktuell"))
    return records


def crawl_stadt_heilbronn():
    records = []
    feed = feedparser.parse("https://www.heilbronn.de/rathaus/aktuelles/pressemitteilungen.xml")

    for entry in feed.entries:
        description = clean_html(entry.get("description", ""))
        if not is_relevant(entry.title, description, "Heilbronn"):
            continue
        records.append(make_record(
            entry.link, entry.get("published", ""), "press_release",
            f"{entry.title}\n\n{description}", "Stadt Heilbronn"))
    return records


def crawl_google_news():
    """Public RSS, no key. Returns headlines and snippets only."""
    records = []
    seen = set()
    editions = [("en", "US", "US:en"), ("en", "GB", "GB:en"), ("de", "DE", "DE:de")]
    queries = {
        "en": ['Heilbronn "cable car"', "Heilbronn gondola", "Heilbronn cableway BUGA"],
        "de": ["Heilbronn Seilbahn", "Heilbronn Gondel Wartberg", "Heilbronn Kabinenbahn"],
    }

    for hl, gl, ceid in editions:
        for q in queries[hl]:
            url = (f"https://news.google.com/rss/search?q={quote_plus(q)}"
                   f"&hl={hl}&gl={gl}&ceid={ceid}")
            for entry in feedparser.parse(url).entries:
                if entry.link in seen:
                    continue
                seen.add(entry.link)

                summary = clean_html(entry.get("summary", ""))
                if not is_relevant(entry.title, summary):
                    continue
                records.append(make_record(
                    entry.link, entry.get("published", ""), "news_article",
                    f"{entry.title}\n\n{summary}",
                    entry.get("source", {}).get("title", "Google News")))
    return records


def write_output(records, destination):
    """Publish the finished snapshot as one rename on the destination filesystem."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with NamedTemporaryFile(mode="w", encoding="utf-8", dir=destination.parent,
                                suffix=".tmp", delete=False) as handle:
            temp_path = Path(handle.name)
            json.dump(records, handle, ensure_ascii=False, indent=4)
            handle.write("\n")
        os.replace(temp_path, destination)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Crawl recent Heilbronn cable-car coverage into a JSON file.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help=f"output JSON path (default: {DEFAULT_OUTPUT})")
    args = parser.parse_args()
    all_data = []

    for label, func in [
        ("SWR Aktuell", crawl_swr),
        ("Stadt Heilbronn", crawl_stadt_heilbronn),
        ("Google News (EN + DE)", crawl_google_news),
    ]:
        print(f"Fetching {label}...")
        all_data.extend(func())

    all_data = list({r["url"]: r for r in all_data}.values())

    by_lang = {}
    for r in all_data:
        by_lang[r["language"]] = by_lang.get(r["language"], 0) + 1
    print(f"Extraction complete. {len(all_data)} relevant records: {by_lang}")

    destination = args.output.expanduser().resolve()
    write_output(all_data, destination)
    print(f"Data successfully exported to {destination}.")


if __name__ == "__main__":
    main()
