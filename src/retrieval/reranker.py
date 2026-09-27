import pandas as pd


FILTER_COLUMNS = {
    "product_group": "product_group_name",
    "product_type": "product_type_name",
    "colour": "colour_group_name",
    "department": "department_name",
    "section": "section_name",
}


def filter_and_rank(
    candidates: pd.DataFrame,
    filters: dict[str, str | None],
    limit: int,
) -> list[dict]:
    result = candidates
    for request_key, column in FILTER_COLUMNS.items():
        value = filters.get(request_key)
        if value:
            result = result[result[column].fillna("").str.casefold() == value.casefold()]
    result = result.sort_values("similarity", ascending=False).head(limit)
    records = result.to_dict(orient="records")
    return [
        {key: None if pd.isna(value) else value for key, value in record.items()}
        for record in records
    ]
