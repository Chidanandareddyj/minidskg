"""Build a typed NetworkX MultiDiGraph and save GraphML."""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import networkx as nx

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
sys.path.insert(0, str(SRC))

from match_mentions import CATALOG_PATH, MENTIONS_PATH, PAPERS_PATH, load_catalog  # noqa: E402

GRAPHML_PATH = ROOT / "data" / "mini_dskg.graphml"


def _nid(kind: str, key: str) -> str:
    return f"{kind}:{key}"


def normalize_author(name: str) -> str:
    return " ".join(name.lower().split())


def load_papers(path: Path = PAPERS_PATH) -> dict[str, dict]:
    papers = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            papers[rec["id"]] = rec
    return papers


def load_mentions(path: Path = MENTIONS_PATH) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def dataset_paper_sets(mentions: list[dict]) -> dict[str, set[str]]:
    papers = defaultdict(set)
    for row in mentions:
        papers[row["dataset_id"]].add(row["paper_id"])
    return papers


def similar_pairs(paper_sets: dict[str, set[str]], top_k: int = 3, min_jaccard: float = 0.05):
    ids = list(paper_sets)
    scores: dict[str, list[tuple[float, str]]] = defaultdict(list)
    for i, a in enumerate(ids):
        sa = paper_sets[a]
        if not sa:
            continue
        for b in ids[i + 1 :]:
            sb = paper_sets[b]
            union = len(sa | sb)
            if not union:
                continue
            j = len(sa & sb) / union
            if j >= min_jaccard:
                scores[a].append((j, b))
                scores[b].append((j, a))
    pairs = []
    seen = set()
    for a, ranked in scores.items():
        ranked.sort(reverse=True)
        for j, b in ranked[:top_k]:
            key = tuple(sorted((a, b)))
            if key in seen:
                continue
            seen.add(key)
            pairs.append((a, b, round(j, 4)))
    return pairs


def build_graph(catalog, papers, mentions) -> nx.MultiDiGraph:
    G = nx.MultiDiGraph()
    linked = {r["dataset_id"] for r in mentions}

    for row in catalog:
        if row["id"] not in linked:
            continue
        G.add_node(
            _nid("dataset", row["id"]),
            kind="Dataset",
            name=row["name"],
            aliases=row["aliases"],
            license=row["license"],
            homepage=row["homepage"],
            year=int(row["year"] or 0),
        )
        task_id = _nid("task", row["task"])
        G.add_node(task_id, kind="Task", name=row["task"])
        G.add_edge(_nid("dataset", row["id"]), task_id, relation="usedFor")

    used_papers = {r["paper_id"] for r in mentions}
    for pid, rec in papers.items():
        if pid not in used_papers:
            continue
        abstract = rec.get("abstract") or ""
        G.add_node(
            _nid("paper", pid),
            kind="Paper",
            name=rec.get("title") or pid,
            title=rec.get("title") or "",
            year=int(rec.get("year") or 0),
            doi=rec.get("doi") or "",
            cited_by_count=int(rec.get("cited_by_count") or 0),
            abstract=abstract[:800],
        )
        for author in rec.get("authors") or []:
            raw = author.get("name") or ""
            key = normalize_author(raw)
            if not key:
                continue
            aid = _nid("author", key)
            if aid not in G:
                G.add_node(aid, kind="Author", name=raw)
            G.add_edge(_nid("paper", pid), aid, relation="writtenBy")

    for row in mentions:
        G.add_edge(
            _nid("paper", row["paper_id"]),
            _nid("dataset", row["dataset_id"]),
            relation="mentioned",
            alias=row["alias"],
            field=row["field"],
        )

    for a, b, j in similar_pairs(dataset_paper_sets(mentions)):
        G.add_edge(_nid("dataset", a), _nid("dataset", b), relation="similarTo", weight=j)
        G.add_edge(_nid("dataset", b), _nid("dataset", a), relation="similarTo", weight=j)
    return G


def main() -> None:
    catalog = load_catalog(CATALOG_PATH)
    papers = load_papers()
    mentions = load_mentions()
    G = build_graph(catalog, papers, mentions)
    GRAPHML_PATH.parent.mkdir(parents=True, exist_ok=True)
    nx.write_graphml(G, GRAPHML_PATH)
    print(
        f"{G.number_of_nodes()} nodes, {G.number_of_edges()} edges -> {GRAPHML_PATH}"
    )


if __name__ == "__main__":
    main()
