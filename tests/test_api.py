import pytest
from pydantic import ValidationError

from api_service.main import ItemScore, RecommendResponse, SimilarityResponse


def test_item_score_valid() -> None:
    item = ItemScore(item_id=123, score=0.95)
    assert item.item_id == 123
    assert item.score == 0.95


def test_recommend_response_valid() -> None:
    response = RecommendResponse(
        visitor_id=101896,
        recommendations=[ItemScore(item_id=123, score=0.95)],
    )
    assert response.visitor_id == 101896
    assert len(response.recommendations) == 1


def test_similarity_response_valid() -> None:
    response = SimilarityResponse(
        item_id=76196,
        similarities=[ItemScore(item_id=153908, score=0.95)],
    )
    assert response.item_id == 76196
    assert len(response.similarities) == 1


def test_item_score_invalid() -> None:
    with pytest.raises(ValidationError):
        ItemScore(item_id="not_an_int", score="not_a_float")
