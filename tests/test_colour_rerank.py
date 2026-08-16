import numpy as np

from aic2026.models import Candidate
from aic2026.reranking.color import (
    colour_fraction,
    _colour_prompt_variants,
    query_colours,
    rerank_with_colour_evidence,
    object_colour_evidence,
    torso_colour_evidence,
)


def test_query_colours_recognizes_normalized_colour_words() -> None:
    assert query_colours("a purple shirt beside a grey car") == {"purple", "gray"}
    assert query_colours("áo màu xanh dương và mũ tím") == {"blue", "purple"}


def test_colour_counterfactuals_only_change_colour_word() -> None:
    variants = _colour_prompt_variants("a man wearing a red shirt", "red")
    assert "a man wearing a blue shirt" in variants
    assert all("red" not in prompt for prompt in variants)


def test_colour_fraction_separates_red_from_blue() -> None:
    red = np.full((20, 20, 3), [255, 0, 0], dtype=np.uint8)
    assert colour_fraction(red, "red") > 0.95
    assert colour_fraction(red, "blue") == 0.0


def test_torso_colour_ignores_a_red_background_outside_person() -> None:
    image = np.full((100, 100, 3), [255, 0, 0], dtype=np.uint8)
    image[30:75, 35:65] = [0, 0, 255]  # detected person's blue shirt/torso
    person = [(0.3, 0.35, 0.75, 0.65)]
    assert torso_colour_evidence(image, "red", person) == 0.0
    assert torso_colour_evidence(image, "blue", person) > 0.9


def test_object_colour_ignores_background_outside_car_box() -> None:
    image = np.full((100, 100, 3), [255, 0, 0], dtype=np.uint8)
    image[30:70, 20:80] = [0, 0, 255]
    car = [(0.3, 0.2, 0.7, 0.8)]
    assert object_colour_evidence(image, "red", car) == 0.0
    assert object_colour_evidence(image, "blue", car) > 0.9


def test_missing_images_do_not_change_ranking() -> None:
    candidates = [
        Candidate(video_id="v1", frame_id=1, score=0.8, vector_id=1, keyframe_path="missing.jpg"),
        Candidate(video_id="v2", frame_id=2, score=0.7, vector_id=2, keyframe_path="missing2.jpg"),
    ]
    assert rerank_with_colour_evidence("red car", candidates) == candidates
