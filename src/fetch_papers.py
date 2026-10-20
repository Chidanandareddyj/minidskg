"""Fetch paper abstracts from OpenAlex for each catalog dataset (MAG's successor)."""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

socket.setdefaulttimeout(12)

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))

from match_mentions import CATALOG_PATH, load_catalog  # noqa: E402

PAPERS_PATH = ROOT / "data" / "papers.jsonl"
OPENALEX = "https://api.openalex.org/works"
UA = "Mini-DSKG/0.1 (student lab; https://openalex.org)"


def reconstruct_abstract(inverted: dict | None) -> str:
    if not inverted:
        return ""
    positions = [p for ps in inverted.values() for p in ps]
    if not positions:
        return ""
    words = [""] * (max(positions) + 1)
    for word, ps in inverted.items():
        for p in ps:
            words[p] = word
    return " ".join(words)


def openalex_get(url: str, retries: int = 1) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    delay = 1.0
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=12) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503) and attempt < retries - 1:
                wait = 5.0 if e.code == 429 else delay
                time.sleep(wait)
                delay *= 2
                continue
            raise
        except urllib.error.URLError:
            if attempt < retries - 1:
                time.sleep(delay)
                delay *= 2
                continue
            raise
    raise RuntimeError("unreachable")


def search_dataset(name: str, per_page: int) -> list[dict]:
    params = {
        "search": name,
        "filter": "has_abstract:true",
        "per_page": str(per_page),
        "select": "id,doi,display_name,publication_year,cited_by_count,abstract_inverted_index,authorships",
    }
    mailto = os.environ.get("OPENALEX_MAILTO", "").strip()
    if mailto:
        params["mailto"] = mailto
    url = OPENALEX + "?" + urllib.parse.urlencode(params)
    payload = openalex_get(url)
    out = []
    for work in payload.get("results") or []:
        authors = []
        for a in work.get("authorships") or []:
            author = a.get("author") or {}
            display = (author.get("display_name") or "").strip()
            if display:
                authors.append({"id": author.get("id") or "", "name": display})
        openalex_id = (work.get("id") or "").rsplit("/", 1)[-1]
        out.append(
            {
                "id": openalex_id,
                "doi": work.get("doi") or "",
                "title": work.get("display_name") or "",
                "year": work.get("publication_year") or 0,
                "cited_by_count": work.get("cited_by_count") or 0,
                "abstract": reconstruct_abstract(work.get("abstract_inverted_index")),
                "authors": authors,
            }
        )
    time.sleep(0.4)
    return out


ATOM = {"a": "http://www.w3.org/2005/Atom"}


def search_arxiv(name: str, per_page: int) -> list[dict]:
    """Fallback when OpenAlex rate-limits. Abstracts only; citation counts stay 0."""
    qs = urllib.parse.urlencode(
        {"search_query": f'all:"{name}"', "start": 0, "max_results": per_page}
    )
    url = "https://export.arxiv.org/api/query?" + qs
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=12) as resp:
        root = ET.fromstring(resp.read())
    out = []
    for entry in root.findall("a:entry", ATOM):
        aid = (entry.findtext("a:id", default="", namespaces=ATOM) or "").rsplit("/", 1)[-1]
        title = " ".join((entry.findtext("a:title", default="", namespaces=ATOM) or "").split())
        abstract = " ".join((entry.findtext("a:summary", default="", namespaces=ATOM) or "").split())
        published = entry.findtext("a:published", default="", namespaces=ATOM) or ""
        year = int(published[:4]) if published[:4].isdigit() else 0
        authors = []
        for au in entry.findall("a:author", ATOM):
            nm = (au.findtext("a:name", default="", namespaces=ATOM) or "").strip()
            if nm:
                authors.append({"id": "", "name": nm})
        if not aid or not title:
            continue
        out.append(
            {
                "id": "arxiv:" + aid,
                "doi": "",
                "title": title,
                "year": year,
                "cited_by_count": 0,
                "abstract": abstract,
                "authors": authors,
            }
        )
    time.sleep(3)
    return out


def merge_papers(existing: dict[str, dict], batch: list[dict], query_dataset: str) -> None:
    for paper in batch:
        prev = existing.get(paper["id"])
        if prev is None:
            paper["query_datasets"] = query_dataset
            existing[paper["id"]] = paper
            continue
        seen = set(prev.get("query_datasets", "").split("|")) - {""}
        seen.add(query_dataset)
        prev["query_datasets"] = "|".join(sorted(seen))


def load_existing(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    found = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        found[rec["id"]] = rec
    return found


def write_jsonl(papers: dict[str, dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in papers.values():
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def fetch_all(per_page: int = 25, max_datasets: int = 0, workers: int = 1) -> dict[str, dict]:
    catalog = load_catalog(CATALOG_PATH)
    if max_datasets:
        catalog = catalog[:max_datasets]
    papers = load_existing(PAPERS_PATH)
    already = set()
    for rec in papers.values():
        already.update(rec.get("query_datasets", "").split("|"))
    already.discard("")
    todo = [row for row in catalog if row["id"] not in already]
    print(f"cached {len(already)} datasets, fetching {len(todo)}", flush=True)

    def one(row):
        print(f"fetch {row['name']}", flush=True)
        try:
            batch = search_dataset(row["name"], per_page=per_page)
        except Exception as e:
            print(f"OpenAlex miss {row['name']}: {e}", flush=True)
            batch = []
        if not batch:
            try:
                batch = search_arxiv(row["name"], per_page=min(per_page, 20))
                print(f"  arXiv fallback {row['name']}: {len(batch)}", flush=True)
            except Exception as e:
                print(f"skip {row['name']}: {e}", flush=True)
                batch = []
        return row["id"], row["name"], batch

    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(one, row) for row in todo]
        for fut in as_completed(futs):
            dataset_id, name, batch = fut.result()
            merge_papers(papers, batch, dataset_id)
            write_jsonl(papers, PAPERS_PATH)
            done += 1
            print(f"[{done}/{len(todo)}] {name}: +{len(batch)} papers ({len(papers)} unique)", flush=True)
    return papers


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--per-page", type=int, default=25)
    p.add_argument("--max-datasets", type=int, default=0)
    args = p.parse_args()
    papers = fetch_all(per_page=args.per_page, max_datasets=args.max_datasets)
    print(f"cached {len(papers)} unique papers -> {PAPERS_PATH}")


if __name__ == "__main__":
    main()
