import pandas as pd

from src.retrieval.reranker import filter_and_rank


def test_filter_and_rank_limits_and_filters():
    candidates = pd.DataFrame([
        {"article_id": "1", "product_group_name": "Garment Upper body", "product_type_name": "T-shirt", "colour_group_name": "Black", "department_name": "Ladieswear", "section_name": "Ladies Other", "detail_desc": float("nan"), "similarity": .9},
        {"article_id": "2", "product_group_name": "Garment Upper body", "product_type_name": "Sweater", "colour_group_name": "Black", "department_name": "Ladieswear", "section_name": "Ladies Other", "similarity": .8},
    ])
    result = filter_and_rank(candidates, {"product_type": "T-shirt"}, 1)
    assert [row["article_id"] for row in result] == ["1"]
    assert result[0]["detail_desc"] is None
