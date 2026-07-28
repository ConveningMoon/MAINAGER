from __future__ import annotations

from mainager.cli import _summarise_models


def test_summary_counts_models_inside_each_type() -> None:
    payload = {
        "models": {
            "video": {"grok-itv-10": {}, "veo3_fast": {}},
            "image": {"nano-banana-2-lite": {}},
        }
    }

    assert _summarise_models(payload) == "3 models (image 1, video 2)"


def test_summary_tolerates_a_catalog_without_models() -> None:
    assert _summarise_models({"status": "ok"}) == "no model catalog in payload"
