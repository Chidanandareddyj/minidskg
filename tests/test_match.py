"""One runnable check: CIFAR-10 matches, the stopword 'data' does not."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from match_mentions import find_mentions, load_catalog, usable_alias


def test_cifar_and_imagenet_match():
    catalog = load_catalog()
    text = (
        "We pre-train ResNet-18 on CIFAR-10 and then fine-tune on ImageNet. "
        "This paper presents data for the experiment."
    )
    hits = {dataset_id: alias for dataset_id, alias in find_mentions(text, catalog)}
    assert "cifar-10" in hits, hits
    assert "imagenet" in hits, hits
    assert "data" not in {a.lower() for a in hits.values()}


def test_stopword_alias_dropped():
    assert not usable_alias("data")
    assert not usable_alias("dataset")
    catalog = [{"id": "dummy", "name": "data", "aliases": "data|dataset"}]
    hits = find_mentions("This paper presents data for the experiment.", catalog)
    assert hits == []


if __name__ == "__main__":
    test_stopword_alias_dropped()
    test_cifar_and_imagenet_match()
    print("test_match.py: ok")
