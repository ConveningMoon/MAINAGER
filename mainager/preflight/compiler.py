"""Turn an Intent into a request body that the chosen model actually accepts.

The interesting case is a source image. Providers spell that parameter at least
six different ways, and sending the wrong one is documented as the most common
agent error: the video model ignores it and the account is still charged. The
compiler never guesses. It reads which of the known media parameters the chosen
model declares and emits that one, or refuses.

`strict` is always set. It is the flag that makes the provider reject an
incompatible body before the debit rather than after, and there is no reason to
ever leave it off.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from mainager.preflight.registry import ModelSpec

MediaType = Literal["image", "video", "voice", "music", "text"]

#: Parameters that carry a source image, most specific first. Which one a given
#: model wants is read from the catalog, never assumed.
IMAGE_SOURCE_PARAMS: tuple[str, ...] = (
    "image_input",
    "image_urls",
    "first_frame_url",
    "reference_image_urls",
    "character_image_url",
    "image_url",
)

#: Parameters whose value is a list of URLs rather than a single URL.
_LIST_VALUED: frozenset[str] = frozenset({"image_input", "image_urls", "reference_image_urls"})

_IMAGE_TO_VIDEO = "image-to-video"


class Intent(BaseModel):
    """What the caller wants, independent of any provider's vocabulary."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    media_type: MediaType
    prompt: str
    source_image_url: str | None = None
    duration_s: int | None = None
    aspect_ratio: str | None = None
    resolution: str | None = None
    idempotency_key: str | None = None
    extra_params: dict[str, Any] = Field(default_factory=dict)


class IncompatibleIntentError(ValueError):
    """The chosen model cannot serve this intent.

    Raised locally, before any network call, so it costs nothing.
    """

    def __init__(self, model_id: str, reason: str) -> None:
        super().__init__(f"{model_id}: {reason}")
        self.model_id = model_id
        self.reason = reason


def image_source_param(spec: ModelSpec) -> str | None:
    """Which parameter this model uses for a source image, if any.

    Required parameters win over optional ones; otherwise the order of
    ``IMAGE_SOURCE_PARAMS`` decides.
    """
    candidates = [p for p in IMAGE_SOURCE_PARAMS if spec.accepts(p)]
    if not candidates:
        return None
    for param in candidates:
        if param in spec.required:
            return param
    return candidates[0]


def _check_enum(spec: ModelSpec, param: str, value: str) -> None:
    allowed = spec.allowed_values(param)
    if allowed is not None and value not in allowed:
        raise IncompatibleIntentError(
            spec.model_id,
            f"{param}={value!r} not supported; catalog allows {', '.join(allowed)}",
        )


def _check_mutual_exclusion(spec: ModelSpec, body: dict[str, Any]) -> None:
    present = {k for k in body if k not in {"type", "model", "prompt", "strict"}}
    groups = spec.mutually_exclusive
    hit = [g for g in groups if g & present]
    if len(hit) > 1:
        collide = sorted(p for g in hit for p in (g & present))
        raise IncompatibleIntentError(
            spec.model_id,
            f"catalog marks these as mutually exclusive: {', '.join(collide)}",
        )


def compile_request(intent: Intent, spec: ModelSpec) -> dict[str, Any]:
    """Build the `/generate` body for this intent against this model.

    Raises ``IncompatibleIntentError`` rather than emitting a body the provider
    would charge for and ignore.
    """
    if spec.media_type != intent.media_type:
        raise IncompatibleIntentError(
            spec.model_id,
            f"model produces {spec.media_type}, intent asks for {intent.media_type}",
        )

    body: dict[str, Any] = {
        "type": intent.media_type,
        "model": spec.model_id,
        "prompt": intent.prompt,
        "strict": True,
    }

    prompt_max = spec.limit("prompt_max")
    if isinstance(prompt_max, int) and len(intent.prompt) > prompt_max:
        raise IncompatibleIntentError(
            spec.model_id,
            f"prompt is {len(intent.prompt)} chars, model caps it at {prompt_max}",
        )

    if intent.source_image_url is not None:
        param = image_source_param(spec)
        if param is None:
            raise IncompatibleIntentError(
                spec.model_id,
                "takes no source image; sending one would be ignored and still billed",
            )
        body[param] = (
            [intent.source_image_url] if param in _LIST_VALUED else intent.source_image_url
        )

        max_items = spec.limit(param)
        if param in _LIST_VALUED and isinstance(max_items, int) and max_items < 1:
            raise IncompatibleIntentError(spec.model_id, f"{param} accepts {max_items} items")

        # veo-family models need to be told this is image-to-video; the catalog
        # exposes that as an enum rather than as prose.
        gen_type = spec.allowed_values("generation_type")
        if gen_type and _IMAGE_TO_VIDEO in gen_type:
            body["generation_type"] = _IMAGE_TO_VIDEO

    if intent.aspect_ratio is not None:
        if not spec.accepts("aspect_ratio"):
            raise IncompatibleIntentError(spec.model_id, "does not accept aspect_ratio")
        _check_enum(spec, "aspect_ratio", intent.aspect_ratio)
        body["aspect_ratio"] = intent.aspect_ratio

    if intent.resolution is not None:
        if not spec.accepts("resolution"):
            raise IncompatibleIntentError(spec.model_id, "does not accept resolution")
        _check_enum(spec, "resolution", intent.resolution)
        body["resolution"] = intent.resolution

    if intent.duration_s is not None:
        if not spec.accepts("duration"):
            raise IncompatibleIntentError(spec.model_id, "does not accept duration")
        _check_duration(spec, intent.duration_s)
        body["duration"] = intent.duration_s

    for key, value in intent.extra_params.items():
        if not spec.accepts(key):
            raise IncompatibleIntentError(
                spec.model_id, f"does not accept {key!r}; it would be dropped and still billed"
            )
        body[key] = value

    if intent.idempotency_key is not None:
        body["idempotency_key"] = intent.idempotency_key

    missing = [p for p in spec.required if p not in body]
    if missing:
        raise IncompatibleIntentError(
            spec.model_id, f"missing required parameter(s): {', '.join(missing)}"
        )

    _check_mutual_exclusion(spec, body)
    return body


def _check_duration(spec: ModelSpec, duration_s: int) -> None:
    """Validate against the catalog's ``"4-15s"`` style duration range."""
    raw = spec.limit("duration")
    if not isinstance(raw, str) or "-" not in raw:
        return
    low, _, high = raw.partition("-")
    try:
        low_v = int(low.strip().rstrip("s"))
        high_v = int(high.strip().rstrip("s"))
    except ValueError:
        return
    if not low_v <= duration_s <= high_v:
        raise IncompatibleIntentError(
            spec.model_id, f"duration {duration_s}s outside supported range {raw}"
        )
