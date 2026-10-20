"""Centrality, communities, dataset h-index, Jaccard similarity, problem search."""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import networkx as nx

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
sys.path.insert(0, str(SRC))

from build_graph import GRAPHML_PATH  # noqa: E402

OUT_PATH = ROOT / "outputs" / "analysis.json"


def load_graph(path: Path = GRAPHML_PATH) -> nx.MultiDiGraph:
    return nx.read_graphml(path)


def nodes_of(G: nx.MultiDiGraph, kind: str) -> list[str]:
    return [n for n, d in G.nodes(data=True) if d.get("kind") == kind]


def _int(v, default=0) -> int:
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return default


def mention_counts(G: nx.MultiDiGraph) -> dict[str, int]:
    counts = defaultdict(int)
    for u, v, d in G.edges(data=True):
        if d.get("relation") == "mentioned":
            counts[v] += 1
    return dict(counts)


def paper_neighbors(G: nx.MultiDiGraph, dataset_id: str) -> list[str]:
    return [
        u
        for u, v, d in G.in_edges(dataset_id, data=True)
        if d.get("relation") == "mentioned" and G.nodes[u].get("kind") == "Paper"
    ]


def dataset_h_index(G: nx.MultiDiGraph, dataset_id: str) -> int:
    cites = sorted(
        (_int(G.nodes[p].get("cited_by_count")) for p in paper_neighbors(G, dataset_id)),
        reverse=True,
    )
    h = 0
    for i, c in enumerate(cites, 1):
        if c >= i:
            h = i
        else:
            break
    return h


def dataset_projection(G: nx.MultiDiGraph) -> nx.Graph:
    """Undirected dataset graph: edge if two datasets share a paper."""
    P = nx.Graph()
    for n in nodes_of(G, "Dataset"):
        P.add_node(n, name=G.nodes[n].get("name", n))
    papers: dict[str, list[str]] = defaultdict(list)
    for u, v, d in G.edges(data=True):
        if d.get("relation") == "mentioned":
            papers[u].append(v)
    for ds_list in papers.values():
        uniq = list(dict.fromkeys(ds_list))
        for i, a in enumerate(uniq):
            for b in uniq[i + 1 :]:
                if P.has_edge(a, b):
                    P[a][b]["weight"] += 1
                else:
                    P.add_edge(a, b, weight=1)
    return P


def jaccard(G: nx.MultiDiGraph, a: str, b: str) -> float:
    sa, sb = set(paper_neighbors(G, a)), set(paper_neighbors(G, b))
    union = len(sa | sb)
    return (len(sa & sb) / union) if union else 0.0


def similar_to(G: nx.MultiDiGraph, dataset_id: str, k: int = 5) -> list[tuple[str, float]]:
    scored = []
    for other in nodes_of(G, "Dataset"):
        if other == dataset_id:
            continue
        j = jaccard(G, dataset_id, other)
        if j:
            scored.append((other, j))
    scored.sort(key=lambda t: t[1], reverse=True)
    return scored[:k]


def problem_search(G: nx.MultiDiGraph, query: str, top: int = 8) -> list[tuple[str, float]]:
    q = set(re.findall(r"[a-z0-9]+", query.lower()))
    q -= {"the", "a", "an", "of", "for", "in", "on", "to", "and"}
    scores = defaultdict(float)
    for n, d in G.nodes(data=True):
        if d.get("kind") != "Paper":
            continue
        text = f"{d.get('title', '')} {d.get('abstract', '')}".lower()
        words = set(re.findall(r"[a-z0-9]+", text))
        overlap = len(q & words)
        if not overlap:
            continue
        for _, ds, ed in G.out_edges(n, data=True):
            if ed.get("relation") == "mentioned":
                scores[ds] += overlap
    return sorted(scores.items(), key=lambda t: t[1], reverse=True)[:top]


def by_name(G: nx.MultiDiGraph, needle: str) -> str | None:
    needle = needle.lower()
    for n, d in G.nodes(data=True):
        if d.get("kind") == "Dataset" and needle in (d.get("name") or "").lower():
            return n
        if n.lower().endswith(":" + needle.replace(" ", "-")):
            return n
    return None


def analyze(G: nx.MultiDiGraph) -> dict:
    counts = mention_counts(G)
    proj = dataset_projection(G)
    between = nx.betweenness_centrality(proj, weight="weight") if proj.number_of_edges() else {}
    degree_c = nx.degree_centrality(proj) if proj.number_of_nodes() else {}
    closeness = (
        nx.closeness_centrality(proj) if nx.is_connected(proj) and proj.number_of_nodes() > 1 else {}
    )
    communities = []
    if proj.number_of_edges():
        communities = [
            sorted(G.nodes[n].get("name", n) for n in c)
            for c in nx.community.greedy_modularity_communities(proj, weight="weight")
        ]

    ranked = sorted(counts, key=counts.get, reverse=True)
    top_mentions = [
        {"id": n, "name": G.nodes[n].get("name"), "mentions": counts[n], "h_index": dataset_h_index(G, n)}
        for n in ranked[:15]
    ]
    h_ranked = sorted(nodes_of(G, "Dataset"), key=lambda n: dataset_h_index(G, n), reverse=True)
    top_h = [
        {"id": n, "name": G.nodes[n].get("name"), "h_index": dataset_h_index(G, n), "mentions": counts.get(n, 0)}
        for n in h_ranked[:10]
    ]
    top_bridge = [
        {"id": n, "name": G.nodes[n].get("name"), "betweenness": round(between[n], 4)}
        for n in sorted(between, key=between.get, reverse=True)[:10]
    ]

    imagenet = by_name(G, "imagenet")
    similar = []
    impact = []
    if imagenet:
        similar = [
            {"id": n, "name": G.nodes[n].get("name"), "jaccard": round(j, 3)}
            for n, j in similar_to(G, imagenet)
        ]
        # Tutorial subway example: descendants of a removed node, on the
        # directed usedFor/similarTo/mentioned mix — papers that mention it.
        impact = [
            {"id": p, "name": G.nodes[p].get("name")}
            for p in paper_neighbors(G, imagenet)[:8]
        ]

    query = "object detection in autonomous driving"
    search_hits = [
        {"id": n, "name": G.nodes[n].get("name"), "score": s}
        for n, s in problem_search(G, query)
    ]

    return {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "n_datasets": len(nodes_of(G, "Dataset")),
        "n_papers": len(nodes_of(G, "Paper")),
        "n_authors": len(nodes_of(G, "Author")),
        "n_tasks": len(nodes_of(G, "Task")),
        "projection_connected": bool(proj.number_of_nodes()) and nx.is_connected(proj),
        "n_communities": len(communities),
        "communities": communities[:8],
        "top_mentions": top_mentions,
        "top_h_index": top_h,
        "top_betweenness": top_bridge,
        "degree_centrality_sample": {
            G.nodes[n].get("name", n): round(degree_c[n], 3)
            for n in ranked[:8]
            if n in degree_c
        },
        "closeness_available": bool(closeness),
        "similar_to_imagenet": similar,
        "problem_search": {"query": query, "hits": search_hits},
        "papers_using_imagenet_sample": impact,
    }


def main() -> None:
    G = load_graph()
    result = analyze(G)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"{result['n_datasets']} datasets, {result['n_papers']} papers, {result['edges']} edges")
    print("\nTop datasets by mentions")
    for row in result["top_mentions"][:10]:
        print(f"  {row['mentions']:4d}  h={row['h_index']:3d}  {row['name']}")
    print("\nBridge datasets (betweenness on dataset projection)")
    for row in result["top_betweenness"][:8]:
        print(f"  {row['betweenness']:.4f}  {row['name']}")
    print("\nCommunities:", result["n_communities"])
    for i, c in enumerate(result["communities"][:5], 1):
        print(f"  {i}: {', '.join(c[:8])}")
    print("\nSimilar to ImageNet:", result["similar_to_imagenet"])
    print("\nProblem search:", result["problem_search"])
    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
