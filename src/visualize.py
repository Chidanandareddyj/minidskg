"""PyVis explorer: datasets, tasks, and the papers that connect them."""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

from pyvis.network import Network

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
sys.path.insert(0, str(SRC))

from analyze import load_graph, mention_counts, nodes_of  # noqa: E402

HTML_PATH = ROOT / "outputs" / "explorer.html"
COLORS = {
    "Dataset": "#4C78A8",
    "Paper": "#F58518",
    "Author": "#54A24B",
    "Task": "#E45756",
}


def viz_nodes(G):
    counts = mention_counts(G)
    keep = set(nodes_of(G, "Dataset")) | set(nodes_of(G, "Task"))
    papers_by_ds = defaultdict(list)
    multi = []
    for n in nodes_of(G, "Paper"):
        ds = [
            v
            for _, v, d in G.out_edges(n, data=True)
            if d.get("relation") == "mentioned"
        ]
        if len(set(ds)) >= 2:
            multi.append(n)
        for d in set(ds):
            papers_by_ds[d].append(n)
    keep.update(multi)
    for ds, papers in papers_by_ds.items():
        papers.sort(key=lambda p: int(float(G.nodes[p].get("cited_by_count") or 0)), reverse=True)
        keep.update(papers[:3])
    authors = []
    for p in multi:
        for _, a, d in G.out_edges(p, data=True):
            if d.get("relation") == "writtenBy":
                authors.append(a)
    authors.sort(key=lambda a: G.degree(a), reverse=True)
    keep.update(authors[:80])
    return keep, counts


def build_network(G) -> Network:
    keep, counts = viz_nodes(G)
    net = Network(height="800px", width="100%", directed=True, cdn_resources="in_line")
    net.barnes_hut()
    for n in keep:
        d = G.nodes[n]
        kind = d.get("kind", "?")
        label = d.get("name") or n
        title = f"{kind}: {label}"
        if kind == "Dataset":
            title += f"<br>mentions={counts.get(n, 0)}<br>{d.get('homepage', '')}"
            size = 12 + min(counts.get(n, 1), 40)
        elif kind == "Paper":
            title += f"<br>{d.get('year', '')} · cited {d.get('cited_by_count', 0)}"
            size = 8
        elif kind == "Task":
            size = 14
        else:
            size = 6
        net.add_node(n, label=label[:48], title=title, color=COLORS.get(kind, "#bbb"), size=size)
    for u, v, d in G.edges(data=True):
        if u not in keep or v not in keep:
            continue
        rel = d.get("relation", "")
        if rel == "similarTo":
            continue
        net.add_edge(u, v, title=rel, arrows="to")
    net.set_options(
        """
        {"interaction": {"hover": true},
         "nodes": {"font": {"size": 12}},
         "edges": {"color": {"opacity": 0.4}, "smooth": false}}
        """
    )
    return net


def main() -> None:
    G = load_graph()
    net = build_network(G)
    HTML_PATH.parent.mkdir(parents=True, exist_ok=True)
    html = net.generate_html(notebook=False)
    HTML_PATH.write_text(html, encoding="utf-8")
    print(f"wrote {HTML_PATH}")


if __name__ == "__main__":
    main()
