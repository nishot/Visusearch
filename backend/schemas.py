from pydantic import BaseModel, Field


class ProductResult(BaseModel):
    article_id: str
    prod_name: str | None = None
    product_type_name: str | None = None
    product_group_name: str | None = None
    colour_group_name: str | None = None
    department_name: str | None = None
    section_name: str | None = None
    detail_desc: str | None = None
    image_path: str
    similarity: float


class SearchResponse(BaseModel):
    results: list[ProductResult]
    count: int


class SearchFilters(BaseModel):
    limit: int = Field(default=20, ge=1, le=100)
    product_group: str | None = None
    product_type: str | None = None
    colour: str | None = None
    department: str | None = None
    section: str | None = None
