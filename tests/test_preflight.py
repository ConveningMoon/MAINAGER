"""Golden set: everything here runs against the committed catalog snapshot.

If the provider changes a model's contract, refreshing data/capabilities.json
makes these tests fail, which is the intended alarm.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mainager.preflight.compiler import (
    IncompatibleIntentError,
    Intent,
    compile_request,
    image_source_param,
)
from mainager.preflight.registry import ModelRegistry, UnknownModelError
from mainager.providers.vibemarketolog.capabilities import read_snapshot

SNAPSHOT_DIR = Path(__file__).resolve().parent.parent / "data"


@pytest.fixture(scope="module")
def registry() -> ModelRegistry:
    return ModelRegistry.from_capabilities(read_snapshot(SNAPSHOT_DIR))


def test_registry_covers_every_catalog_type(registry: ModelRegistry) -> None:
    for media_type in ("image", "video", "voice", "music"):
        assert registry.by_type(media_type), f"no models of type {media_type}"


def test_tiers_are_callable_models_with_their_own_price(registry: ModelRegistry) -> None:
    """`model: grok-itv-10` is a real request value, not a parameter on grok-itv."""
    parent = registry.get("grok-itv")
    tier = registry.get("grok-itv-10")

    assert tier.is_tier_of == "grok-itv"
    assert tier.price == 196
    assert parent.price == 36
    assert tier.required == parent.required


def test_unknown_model_names_itself_in_the_error(registry: ModelRegistry) -> None:
    with pytest.raises(UnknownModelError) as excinfo:
        registry.get("gpt-5-video")
    assert "gpt-5-video" in str(excinfo.value)


# --- the trap table, derived rather than transcribed -----------------------

EXPECTED_IMAGE_PARAM = {
    "grok-itv": "image_urls",
    "kling-3.0-std": "image_urls",
    "kling-3.0-pro": "image_urls",
    "veo3": "image_urls",
    "veo3.1": "image_urls",
    "veo3_fast": "image_urls",
    "gemini-omni-video": "image_urls",
    "seedance-2": "first_frame_url",
    "seedance-2-fast": "first_frame_url",
    "seedance-2-mini": "first_frame_url",
    "motion-control-720p": "character_image_url",
    "motion-control-1080p": "character_image_url",
    "nano-banana-2-lite": "image_input",
    "seedream-5-pro-edit": "image_input",
    "gpt-image-2-edit": "image_input",
}


@pytest.mark.parametrize(("model_id", "expected"), sorted(EXPECTED_IMAGE_PARAM.items()))
def test_source_image_param_matches_the_catalog(
    registry: ModelRegistry, model_id: str, expected: str
) -> None:
    assert image_source_param(registry.get(model_id)) == expected


def test_models_without_a_source_image_param_are_recognised(registry: ModelRegistry) -> None:
    for model_id in ("grok-ttv", "z-image", "seedream-5-pro"):
        assert image_source_param(registry.get(model_id)) is None


# --- compilation ------------------------------------------------------------


def test_image_to_video_emits_the_field_the_model_declares(registry: ModelRegistry) -> None:
    intent = Intent(
        media_type="video", prompt="a cat walking", source_image_url="https://cdn.example/a.png"
    )

    assert compile_request(intent, registry.get("grok-itv"))["image_urls"] == [
        "https://cdn.example/a.png"
    ]
    assert compile_request(intent, registry.get("seedance-2-mini"))["first_frame_url"] == (
        "https://cdn.example/a.png"
    )


def test_motion_control_needs_its_reference_video_too(registry: ModelRegistry) -> None:
    """The catalog marks both character_image_url and reference_video_url required."""
    image_only = Intent(
        media_type="video", prompt="dance", source_image_url="https://cdn.example/a.png"
    )

    with pytest.raises(IncompatibleIntentError) as excinfo:
        compile_request(image_only, registry.get("motion-control-720p"))
    assert "reference_video_url" in str(excinfo.value)

    complete = image_only.model_copy(
        update={"extra_params": {"reference_video_url": "https://cdn.example/v.mp4"}}
    )
    body = compile_request(complete, registry.get("motion-control-720p"))
    assert body["character_image_url"] == "https://cdn.example/a.png"
    assert body["reference_video_url"] == "https://cdn.example/v.mp4"


def test_the_documented_money_burning_mistake_is_refused_locally(
    registry: ModelRegistry,
) -> None:
    """image_input to a text-to-video model: ignored by the provider, still billed."""
    intent = Intent(
        media_type="video", prompt="a cat walking", source_image_url="https://cdn.example/a.png"
    )

    with pytest.raises(IncompatibleIntentError) as excinfo:
        compile_request(intent, registry.get("grok-ttv"))

    assert "still billed" in str(excinfo.value)


def test_veo_family_is_told_it_is_image_to_video(registry: ModelRegistry) -> None:
    intent = Intent(
        media_type="video", prompt="a cat", source_image_url="https://cdn.example/a.png"
    )
    body = compile_request(intent, registry.get("veo3_fast"))
    assert body["generation_type"] == "image-to-video"


def test_text_to_video_does_not_get_a_generation_type(registry: ModelRegistry) -> None:
    body = compile_request(Intent(media_type="video", prompt="a cat"), registry.get("veo3_fast"))
    assert "generation_type" not in body


def test_strict_is_always_set(registry: ModelRegistry) -> None:
    for model_id in ("grok-ttv", "z-image", "suno-v5.5", "el-tts-turbo"):
        spec = registry.get(model_id)
        body = compile_request(Intent(media_type=spec.media_type, prompt="x"), spec)  # type: ignore[arg-type]
        assert body["strict"] is True


def test_unsupported_aspect_ratio_is_rejected_with_the_allowed_list(
    registry: ModelRegistry,
) -> None:
    intent = Intent(media_type="image", prompt="a poster", aspect_ratio="16:9")

    with pytest.raises(IncompatibleIntentError) as excinfo:
        compile_request(intent, registry.get("nano-banana-2-lite"))

    message = str(excinfo.value)
    assert "16:9" in message
    assert "4:5" in message  # the catalog's allowed set is reported back


def test_supported_aspect_ratio_passes_through(registry: ModelRegistry) -> None:
    intent = Intent(media_type="image", prompt="a poster", aspect_ratio="4:5")
    assert compile_request(intent, registry.get("nano-banana-2-lite"))["aspect_ratio"] == "4:5"


def test_duration_outside_the_catalog_range_is_rejected(registry: ModelRegistry) -> None:
    with pytest.raises(IncompatibleIntentError) as excinfo:
        compile_request(
            Intent(media_type="video", prompt="a cat", duration_s=2),
            registry.get("seedance-2-mini"),
        )
    assert "4-15s" in str(excinfo.value)


def test_prompt_over_the_model_cap_is_rejected(registry: ModelRegistry) -> None:
    with pytest.raises(IncompatibleIntentError) as excinfo:
        compile_request(Intent(media_type="image", prompt="x" * 1001), registry.get("z-image"))
    assert "1000" in str(excinfo.value)


def test_unknown_extra_param_is_refused_rather_than_silently_dropped(
    registry: ModelRegistry,
) -> None:
    intent = Intent(media_type="video", prompt="a cat", extra_params={"mode": "pro"})

    with pytest.raises(IncompatibleIntentError) as excinfo:
        compile_request(intent, registry.get("grok-ttv"))

    assert "mode" in str(excinfo.value)


def test_catalog_declared_mutual_exclusion_is_enforced(registry: ModelRegistry) -> None:
    intent = Intent(
        media_type="video",
        prompt="a cat",
        source_image_url="https://cdn.example/a.png",
        extra_params={"reference_image_urls": ["https://cdn.example/b.png"]},
    )

    with pytest.raises(IncompatibleIntentError) as excinfo:
        compile_request(intent, registry.get("seedance-2-mini"))

    assert "mutually exclusive" in str(excinfo.value)


def test_type_mismatch_is_caught(registry: ModelRegistry) -> None:
    with pytest.raises(IncompatibleIntentError):
        compile_request(Intent(media_type="image", prompt="x"), registry.get("grok-ttv"))


def test_missing_required_parameter_is_caught(registry: ModelRegistry) -> None:
    """grok-itv requires image_urls; an intent without an image cannot use it."""
    with pytest.raises(IncompatibleIntentError) as excinfo:
        compile_request(Intent(media_type="video", prompt="a cat"), registry.get("grok-itv"))
    assert "image_urls" in str(excinfo.value)
