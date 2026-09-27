from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.schemas import SearchFilters, SearchResponse
from backend.services.visual_search_service import VisualSearchService
from src.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
service: VisualSearchService | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global service
    logger.info("Loading visual search assets on startup")
    service = VisualSearchService()
    yield
    service = None


app = FastAPI(title="H&M Visual Search", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/images", StaticFiles(directory=settings.images_root), name="images")


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(settings.project_root / "frontend" / "index.html")


@app.get("/api/health")
def health():
    return {"status": "ok", "ready": service is not None}


@app.post("/api/search", response_model=SearchResponse)
async def search_image(
    image: UploadFile = File(...),
    limit: int = Form(20, ge=1, le=100),
    product_group: str | None = Form(None),
    product_type: str | None = Form(None),
    colour: str | None = Form(None),
    department: str | None = Form(None),
    section: str | None = Form(None),
):
    if service is None:
        raise HTTPException(status_code=503, detail="Search service is still loading.")
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Upload a JPEG, PNG, or WebP image.")
    payload = await image.read()
    if not payload or len(payload) > settings.max_upload_bytes:
        raise HTTPException(status_code=413, detail="Image is empty or larger than the upload limit.")
    try:
        filters = SearchFilters(
            limit=limit,
            product_group=product_group,
            product_type=product_type,
            colour=colour,
            department=department,
            section=section,
        )
        results = service.search(payload, filters.limit, filters.model_dump(exclude={"limit"}))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return SearchResponse(results=results, count=len(results)) #type: ignore

