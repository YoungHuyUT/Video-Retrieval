"""Small, deterministic colour-evidence reranker.

This is deliberately not an object recogniser.  It gives a bounded bonus only
when a query explicitly names a colour and a decoded keyframe contains that
colour.  It is cheap enough to run on the leading retrieval pool and is a
useful complement to CLIP, whose global embedding often underweights colour.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import unicodedata
import re
from collections.abc import Callable

import numpy as np

from aic2026.models import Candidate, FrameRecord

_COLOURS = frozenset(
    {"red", "orange", "yellow", "green", "blue", "purple", "pink", "brown", "black", "white", "gray", "grey"}
)
_PERSON_WORDS = frozenset({"person", "people", "human", "man", "woman", "boy", "girl", "child"})
_GARMENT_WORDS = frozenset({"shirt", "t-shirt", "tee", "jacket", "coat", "dress", "uniform", "top", "pants", "trousers"})
_OBJECT_LABELS = {
    "person": _PERSON_WORDS | {"human face"},
    "car": frozenset({"car", "vehicle", "land vehicle", "automobile"}),
    "ball": frozenset({"ball", "football", "soccer ball"}),
    "dog": frozenset({"dog", "puppy"}),
    "cat": frozenset({"cat", "kitten"}),
    "bird": frozenset({"bird"}),
    "horse": frozenset({"horse"}),
    "bicycle": frozenset({"bicycle", "bike"}),
    "motorcycle": frozenset({"motorcycle", "motorbike"}),
    "boat": frozenset({"boat", "ship"}),
    "train": frozenset({"train"}),
    "bus": frozenset({"bus"}),
}


def query_colours(query: str) -> set[str]:
    """Return explicit English or Vietnamese colour constraints in *query*."""
    normalized = unicodedata.normalize("NFD", query.casefold().replace("đ", "d"))
    normalized = "".join(c for c in normalized if unicodedata.category(c) != "Mn")
    vietnamese_phrases = {
        "xanh duong": "blue", "xanh la cay": "green", "xanh la": "green",
        "mau do": "red", "mau vang": "yellow", "mau den": "black",
        "mau trang": "white", "mau tim": "purple", "mau hong": "pink",
        "mau nau": "brown", "mau xam": "gray", "mau cam": "orange",
    }
    translated = normalized
    for phrase, colour in vietnamese_phrases.items():
        translated = translated.replace(phrase, f" {colour} ")
    words = {word.strip(".,;:!?()[]{}\"'").lower() for word in translated.split()}
    words |= {
        {"do": "red", "vang": "yellow", "den": "black", "trang": "white",
         "tim": "purple", "hong": "pink", "nau": "brown", "xam": "gray",
         "cam": "orange", "xanh": "green"}[word]
        for word in words
        if word in {"do", "vang", "den", "trang", "tim", "hong", "nau", "xam", "cam", "xanh"}
    }
    colours = words & _COLOURS
    if "grey" in colours:
        colours.remove("grey")
        colours.add("gray")
    return colours


def _rgb_to_hsv(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rgb = rgb.astype(np.float32) / 255.0
    maximum = rgb.max(axis=2)
    minimum = rgb.min(axis=2)
    delta = maximum - minimum
    saturation = np.divide(delta, maximum, out=np.zeros_like(delta), where=maximum > 1e-6)
    hue = np.zeros_like(maximum)
    nonzero = delta > 1e-6
    red = nonzero & (maximum == rgb[:, :, 0])
    green = nonzero & (maximum == rgb[:, :, 1])
    blue = nonzero & (maximum == rgb[:, :, 2])
    hue[red] = ((rgb[:, :, 1][red] - rgb[:, :, 2][red]) / delta[red]) % 6
    hue[green] = (rgb[:, :, 2][green] - rgb[:, :, 0][green]) / delta[green] + 2
    hue[blue] = (rgb[:, :, 0][blue] - rgb[:, :, 1][blue]) / delta[blue] + 4
    return hue * 60.0, saturation, maximum


def _hue_between(hue: np.ndarray, low: float, high: float) -> np.ndarray:
    return (hue >= low) & (hue <= high) if low <= high else (hue >= low) | (hue <= high)


def colour_fraction(rgb: np.ndarray, colour: str) -> float:
    """Estimate the visible fraction for one colour in an RGB image."""
    hue, sat, value = _rgb_to_hsv(rgb)
    vivid = sat >= 0.28
    ranges = {
        "red": (345, 18), "orange": (18, 45), "yellow": (45, 70),
        "green": (70, 170), "blue": (185, 260), "purple": (260, 320),
        "pink": (320, 345), "brown": (12, 45),
    }
    if colour in ranges:
        low, high = ranges[colour]
        mask = _hue_between(hue, low, high) & vivid
        if colour == "brown":
            mask &= value < 0.72
    elif colour == "white":
        mask = (sat < 0.16) & (value > 0.72)
    elif colour == "black":
        mask = value < 0.22
    elif colour == "gray":
        mask = (sat < 0.16) & (value >= 0.22) & (value <= 0.72)
    else:
        return 0.0
    return float(mask.mean())


def _needs_person_garment_binding(query: str) -> bool:
    words = {word.strip(".,;:!?()[]{}\"'").lower() for word in query.split()}
    return bool(words & _PERSON_WORDS) and bool(words & _GARMENT_WORDS)


def _query_object_labels(query: str) -> set[str]:
    words = {word.strip(".,;:!?()[]{}\"'").lower() for word in query.split()}
    labels: set[str] = set()
    for aliases in _OBJECT_LABELS.values():
        if words & aliases:
            labels.update(aliases)
    return labels


def _object_boxes(
    object_path: str | None, labels: set[str]
) -> list[tuple[float, float, float, float]]:
    """Read high-confidence OpenImages boxes belonging to the requested object."""
    if not object_path:
        return []
    try:
        import json
        payload = json.loads(Path(object_path).read_text(encoding="utf-8"))
        names = payload.get("detection_class_entities", [])
        scores = payload.get("detection_scores", [])
        boxes = payload.get("detection_boxes", [])
    except (OSError, ValueError, TypeError):
        return []
    selected: list[tuple[float, float, float, float]] = []
    for name, score, box in zip(names, scores, boxes):
        if str(name).casefold() not in labels or float(score) < 0.20 or len(box) != 4:
            continue
        top, left, bottom, right = (float(value) for value in box)
        if bottom > top and right > left:
            selected.append((top, left, bottom, right))
    return selected


def _proposal_boxes(object_path: str | None, limit: int = 3) -> list[tuple[float, float, float, float]]:
    """Return the strongest detector boxes without requiring an object-name map."""
    if not object_path:
        return []
    try:
        import json
        payload = json.loads(Path(object_path).read_text(encoding="utf-8"))
        scored = sorted(
            ((float(score), box) for score, box in zip(
                payload.get("detection_scores", []), payload.get("detection_boxes", [])
            )),
            reverse=True,
        )
    except (OSError, ValueError, TypeError):
        return []
    output: list[tuple[float, float, float, float]] = []
    for score, box in scored:
        if score < 0.20 or len(box) != 4:
            continue
        top, left, bottom, right = (float(value) for value in box)
        if bottom > top and right > left and (bottom - top) * (right - left) >= 0.01:
            output.append((top, left, bottom, right))
        if len(output) >= limit:
            break
    return output


def torso_colour_evidence(rgb: np.ndarray, colour: str, boxes: list[tuple[float, float, float, float]]) -> float | None:
    """Return strongest colour fraction in detected persons' torso regions."""
    height, width = rgb.shape[:2]
    fractions: list[float] = []
    for top, left, bottom, right in boxes:
        # Head and legs are deliberately excluded; shirt/top colour normally
        # occupies the middle of a person box.
        y0 = int(max(0, min(height, (top + (bottom - top) * 0.20) * height)))
        y1 = int(max(0, min(height, (top + (bottom - top) * 0.72) * height)))
        x0 = int(max(0, min(width, (left + (right - left) * 0.12) * width)))
        x1 = int(max(0, min(width, (left + (right - left) * 0.88) * width)))
        crop = rgb[y0:y1, x0:x1]
        if crop.size:
            fractions.append(colour_fraction(crop, colour))
    return max(fractions) if fractions else None


def object_colour_evidence(rgb: np.ndarray, colour: str, boxes: list[tuple[float, float, float, float]]) -> float | None:
    """Return strongest colour fraction inside the detected target object boxes."""
    height, width = rgb.shape[:2]
    fractions: list[float] = []
    for top, left, bottom, right in boxes:
        y0, y1 = int(max(0, top * height)), int(min(height, bottom * height))
        x0, x1 = int(max(0, left * width)), int(min(width, right * width))
        crop = rgb[y0:y1, x0:x1]
        if crop.size:
            fractions.append(colour_fraction(crop, colour))
    return max(fractions) if fractions else None


def _colour_prompt_variants(query: str, colour: str) -> list[str]:
    """Create same-object counterfactual prompts by changing only its colour."""
    if not re.search(rf"(?<!\w){re.escape(colour)}(?!\w)", query, flags=re.IGNORECASE):
        return []
    return [
        re.sub(rf"(?<!\w){re.escape(colour)}(?!\w)", other, query, flags=re.IGNORECASE)
        for other in sorted(_COLOURS - {colour, "grey"})
    ]


def contrastive_clip_colour_rerank(
    query: str,
    candidates: list[Candidate],
    records: dict[int, FrameRecord],
    encode_text: Callable[[str], np.ndarray] | None,
    encode_images: Callable[[list[object]], np.ndarray] | None,
    *,
    weight: float = 0.06,
    top_n: int = 50,
) -> list[Candidate]:
    """Automatically bind colour to detector crops with contrastive CLIP.

    No object label vocabulary is used: each detector box is a candidate crop.
    A crop scores well only if the requested-colour prompt beats the same prompt
    with every other colour substituted in its place.
    """
    colours = query_colours(query)
    if len(colours) != 1 or not candidates or encode_text is None or encode_images is None:
        return candidates
    colour = next(iter(colours))
    alternatives = _colour_prompt_variants(query, colour)
    if not alternatives:
        return candidates
    try:
        prompts = [query, *alternatives]
        text_vectors = np.stack([np.asarray(encode_text(prompt), dtype=np.float32) for prompt in prompts])
        text_vectors /= np.maximum(np.linalg.norm(text_vectors, axis=1, keepdims=True), 1e-12)
        from PIL import Image
    except Exception:  # noqa: BLE001 - heuristic colour rerank remains available
        return candidates

    crop_owners: list[int] = []
    crops: list[object] = []
    for index, candidate in enumerate(candidates[:top_n]):
        record = records.get(candidate.vector_id) if candidate.vector_id is not None else None
        if record is None or not candidate.keyframe_path:
            continue
        boxes = _proposal_boxes(record.object_path)
        if not boxes:
            continue
        try:
            with Image.open(Path(candidate.keyframe_path)) as image:
                image.thumbnail((384, 384))
                rgb = image.convert("RGB")
                width, height = rgb.size
                for top, left, bottom, right in boxes:
                    crop = rgb.crop((int(left * width), int(top * height), int(right * width), int(bottom * height)))
                    if crop.width >= 12 and crop.height >= 12:
                        crops.append(crop.copy())
                        crop_owners.append(index)
        except (OSError, ValueError):
            continue
    if not crops:
        return candidates
    try:
        image_vectors = np.asarray(encode_images(crops), dtype=np.float32)
    except Exception:  # noqa: BLE001
        return candidates
    if image_vectors.ndim != 2 or image_vectors.shape[0] != len(crops) or image_vectors.shape[1] != text_vectors.shape[1]:
        return candidates
    image_vectors /= np.maximum(np.linalg.norm(image_vectors, axis=1, keepdims=True), 1e-12)
    similarities = image_vectors @ text_vectors.T
    margins: dict[int, float] = {}
    for row, owner in enumerate(crop_owners):
        margin = float(similarities[row, 0] - similarities[row, 1:].max())
        margins[owner] = max(margins.get(owner, -np.inf), margin)
    output: list[Candidate] = []
    for index, candidate in enumerate(candidates):
        margin = margins.get(index)
        if margin is None:
            output.append(candidate)
            continue
        # CLIP colour margins tend to be small; clamp avoids a single bad crop
        # overwhelming retrieval while still making the wrong-colour candidate fall.
        adjustment = weight * float(np.clip(margin / 0.08, -1.0, 1.0))
        output.append(candidate.model_copy(update={"score": candidate.score + adjustment}))
    return sorted(output, key=lambda candidate: candidate.score, reverse=True)


@lru_cache(maxsize=4096)
def _image_colour_evidence(path: str, colours: tuple[str, ...]) -> float | None:
    try:
        from PIL import Image
        with Image.open(Path(path)) as image:
            image.thumbnail((192, 192))
            rgb = np.asarray(image.convert("RGB"))
    except (ImportError, OSError, ValueError):
        return None
    if rgb.size == 0:
        return None
    # 12% gives a full score: large enough to reject small compression noise,
    # still tolerant of a coloured object occupying only part of a frame.
    return max(min(colour_fraction(rgb, colour) / 0.12, 1.0) for colour in colours)


def _bound_colour_evidence(
    path: str, object_path: str | None, colours: tuple[str, ...],
    labels: set[str], torso: bool,
) -> float | None:
    try:
        from PIL import Image
        with Image.open(Path(path)) as image:
            image.thumbnail((192, 192))
            rgb = np.asarray(image.convert("RGB"))
    except (ImportError, OSError, ValueError):
        return None
    boxes = _object_boxes(object_path, labels)
    scorer = torso_colour_evidence if torso else object_colour_evidence
    evidence = [scorer(rgb, colour, boxes) for colour in colours]
    usable = [value for value in evidence if value is not None]
    return max(usable) if usable else None


def rerank_with_colour_evidence(
    query: str,
    candidates: list[Candidate],
    records: dict[int, FrameRecord] | None = None,
    *,
    weight: float = 0.04,
    top_n: int = 200,
) -> list[Candidate]:
    """Nudge leading candidates by image colour evidence, without VLM calls.

    Missing images/Pillow leave the candidate untouched.  The negative side is
    intentionally smaller than the positive one: global colour is evidence,
    not proof that the queried object itself has that colour.
    """
    colours = tuple(sorted(query_colours(query)))
    if not colours or weight <= 0 or top_n <= 0:
        return candidates
    bind_to_torso = _needs_person_garment_binding(query)
    object_labels = _query_object_labels(query)
    bind_to_object = bool(object_labels)
    reranked: list[Candidate] = []
    for rank, candidate in enumerate(candidates):
        if rank >= top_n or not candidate.keyframe_path:
            reranked.append(candidate)
            continue
        record = records.get(candidate.vector_id) if records and candidate.vector_id is not None else None
        evidence = _bound_colour_evidence(
            candidate.keyframe_path,
            record.object_path if record else None,
            colours,
            object_labels,
            bind_to_torso,
        ) if bind_to_object else _image_colour_evidence(candidate.keyframe_path, colours)
        if evidence is None:
            reranked.append(candidate)
            continue
        # For a bound request such as "man wearing a red shirt", a detected
        # torso with no red is strong negative evidence.  For a whole-frame
        # colour request it remains a softer penalty because the target can be
        # small or outside the central subject.
        adjustment = (
            weight * evidence
            if evidence >= 0.10
            else -weight * (1.0 if bind_to_object else 0.35)
        )
        reranked.append(candidate.model_copy(update={"score": candidate.score + adjustment}))
    return sorted(reranked, key=lambda candidate: candidate.score, reverse=True)
