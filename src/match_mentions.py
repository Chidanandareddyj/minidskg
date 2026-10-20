"""DSKG-style alias matching of dataset names in paper text."""

from __future__ import annotations

import csv
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = ROOT / "data" / "datasets.csv"
MENTIONS_PATH = ROOT / "data" / "mentions.csv"
PAPERS_PATH = ROOT / "data" / "papers.jsonl"

# Same idea as DSKG match_text_corpus.py / filterwords_matching.py: drop aliases
# that are just common English, so "data" never becomes a dataset hit.
STOPWORDS = frozenset(
    """
    a an the and or of to in on for with from by as at is are was were be been
    this that these those it its we our they their you your not no yes if then
    than so such into over under between among using used use based paper study
    method methods model models data dataset datasets image images video text
    speech train test training testing benchmark corpus set net neural network
    learning deep machine result results experiment experiments approach
    proposed new large small real public available open source task tasks
    classification detection recognition segmentation translation
    """.split()
)


def load_catalog(path: Path = CATALOG_PATH) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def usable_alias(alias: str) -> bool:
    a = alias.strip()
    if len(a) < 3:
        return False
    if a.lower() in STOPWORDS:
        return False
    return True


def _flags(alias: str) -> int:
    letters = [c for c in alias if c.isalpha()]
    # Short all-caps tokens (COCO, GLUE, KITTI) match case-sensitively so
    # "race" / "glue" in ordinary English do not count. Upgrade: per-alias
    # allowlist if a short name is commonly written in lowercase.
    if letters and all(c.isupper() for c in letters) and len(alias) <= 6:
        return 0
    return re.IGNORECASE


def compile_alias_index(catalog: list[dict]) -> list[tuple[str, str, re.Pattern]]:
    """Longer aliases first so 'CIFAR-10' wins over a later shorter form."""
    items: list[tuple[str, str, re.Pattern]] = []
    for row in catalog:
        aliases = [row["name"], *row["aliases"].split("|")]
        seen = set()
        for alias in aliases:
            alias = alias.strip()
            key = alias.lower()
            if not usable_alias(alias) or key in seen:
                continue
            seen.add(key)
            pat = re.compile(r"(?<!\w)" + re.escape(alias) + r"(?!\w)", _flags(alias))
            items.append((row["id"], alias, pat))
    items.sort(key=lambda t: len(t[1]), reverse=True)
    return items


def find_mentions(text: str, catalog: list[dict] | None = None, index=None) -> list[tuple[str, str]]:
    """Return unique (dataset_id, matched_alias) pairs, longest alias kept per id."""
    if not text:
        return []
    idx = index if index is not None else compile_alias_index(catalog or [])
    found: dict[str, str] = {}
    for dataset_id, alias, pat in idx:
        if dataset_id in found:
            continue
        if pat.search(text):
            found[dataset_id] = alias
    return list(found.items())


def mention_field(title: str, abstract: str, dataset_id: str, alias: str, index) -> str | None:
    for did, al, pat in index:
        if did != dataset_id or al != alias:
            continue
        if title and pat.search(title):
            return "title"
        if abstract and pat.search(abstract):
            return "abstract"
    return None


def match_papers(papers: list[dict], catalog: list[dict]) -> list[dict]:
    index = compile_alias_index(catalog)
    rows = []
    for paper in papers:
        title = paper.get("title") or ""
        abstract = paper.get("abstract") or ""
        text = f"{title}\n{abstract}"
        for dataset_id, alias in find_mentions(text, index=index):
            field = mention_field(title, abstract, dataset_id, alias, index) or "abstract"
            rows.append(
                {
                    "paper_id": paper["id"],
                    "dataset_id": dataset_id,
                    "alias": alias,
                    "field": field,
                }
            )
    return rows


def write_mentions(rows: list[dict], path: Path = MENTIONS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["paper_id", "dataset_id", "alias", "field"])
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    import json

    catalog = load_catalog()
    papers = [json.loads(line) for line in PAPERS_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = match_papers(papers, catalog)
    write_mentions(rows)
    linked = {r["dataset_id"] for r in rows}
    print(f"wrote {len(rows)} mentions covering {len(linked)}/{len(catalog)} datasets -> {MENTIONS_PATH}")


if __name__ == "__main__":
    main()
