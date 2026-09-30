# H&M Visual Search

A local visual fashion search application that finds visually similar H&M products from an uploaded image.

The system uses **CLIP** to convert images into embeddings and **FAISS** to perform fast similarity search over precomputed H&M product embeddings.

## How It Works

```text
User uploads image
        ↓
FastAPI backend
        ↓
CLIP image encoder
        ↓
Normalized image embedding
        ↓
FAISS similarity search
        ↓
H&M product metadata
        ↓
Filtering & ranking
        ↓
Search results
        ↓
Frontend product grid
```

Product image embeddings are generated **offline** and reused during searches, so product images do not need to be encoded for every request.

## Tech Stack

* **Python**
* **FastAPI** — REST API and backend
* **CLIP (openai/clip-vit-base-patch32)** — image feature extraction
* **FAISS** — vector similarity search
* **Pandas** — product metadata processing
* **Pydantic** — request/response validation
* **HTML, CSS, JavaScript** — frontend
* **H&M Personalized Fashion Recommendations Dataset** — product catalog

## Dataset

The project uses the H&M Personalized Fashion Recommendations dataset from Kaggle.

[H&M Personalized Fashion Recommendations Dataset](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/data?utm_source=chatgpt.com)

Download the dataset before running the application.

> The application currently runs locally.

## Getting Started

### 1. Clone the repository

```powershell
git clone <repository-url>
cd <project-directory>
```

### 2. Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

### 4. Generate catalog embeddings

Run this once after downloading the dataset:

```powershell
python -m scripts.build_embeddings --batch-size 32
```

The first run downloads the CLIP model:

```text
openai/clip-vit-base-patch32
```

Embedding generation supports checkpoints, so an interrupted run can be resumed.

For development or testing, a smaller catalog can be generated with:

```powershell
python -m scripts.build_embeddings --limit 1000 --batch-size 32
```

### 5. Build the FAISS index

```powershell
python -m scripts.build_index
```

### 6. Start the application

```powershell
uvicorn backend.main:app --reload
```

Open:

```text
http://127.0.0.1:8000
```

## Search Features

The backend supports filtering by:

* Product group
* Product type
* Colour
* Department
* Section

The frontend currently exposes the **product group** filter.

## API

### Health Check

```http
GET /api/health
```

### Visual Search

```http
POST /api/search
```

Accepts an uploaded image and optional search filters, then returns visually similar H&M products with similarity scores.

## Project Structure

```text
├── backend/
│   └── main.py
├── src/
│   ├── image_encoder.py
│   ├── retrieval/
│   │   └── faiss_index.py
│   ├── search.py
│   ├── reranker.py
│   └── schemas.py
├── scripts/
│   ├── build_embeddings.py
│   └── build_index.py
├── embeddings/
├── indexes/
├── dataset/
├── index.html
└── requirements.txt
```

## Performance

* Catalog embeddings are generated offline and reused.
* The CLIP model and FAISS index are loaded once when the API starts.
* `IndexFlatIP` performs exact inner-product similarity search.
* Only the uploaded query image is encoded during a search.
* CPU is supported; CUDA can be used on compatible NVIDIA hardware.

## Current Scope

This version focuses on **image-only visual search**.

Future extensions can include:

* Image + text search
* Personalized recommendations
* More advanced vector indexing
* Production deployment
