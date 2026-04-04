# Card Grading AI

Web app that analyzes photos of trading cards (Pokémon, sports, and similar) with computer vision. **Phase 1** focuses on automatic card detection, perspective correction, rough **centering** metrics, and **approximate** grades based on centering only (not official PSA, BGS, or CGC grades).

## Features (MVP)

- Upload a card image from the browser
- Backend detects the card outline (Canny + largest quadrilateral contour), applies a perspective warp to a top-down view
- Heuristic centering (left/right and top/bottom as percentage splits)
- Estimated PSA-, BGS-, and CGC-style scores derived only from centering (educational / demo use)

Planned later: surface defects, edge/corner analysis, scanner integration, persistent scan history (see `backend/app/ml/` and `backend/app/storage/` stubs).

## Tech stack

| Layer | Technology |
|--------|------------|
| Frontend | React, TypeScript, Vite |
| Backend | FastAPI, Python |
| Vision | OpenCV, NumPy |
| Config | Pydantic Settings (optional `CARDGRADING_*` env vars) |

## Repository layout

```
edgegrade/
├── backend/           # FastAPI app
│   ├── app/
│   │   ├── main.py    # App + CORS + routes
│   │   ├── api/routes/analyze.py   # POST /analyze-card
│   │   ├── services/  # detection, centering, pipeline
│   │   ├── models/    # Pydantic schemas
│   │   ├── ml/        # Hooks for future PyTorch, etc.
│   │   └── storage/   # Repository stub for future SQLite
│   └── requirements.txt
└── frontend/          # Vite + React UI
    └── src/
```

## Prerequisites

- **Python** 3.11+ recommended
- **Node.js** 20+ (for Vite)

## Run the backend

From the `backend` directory:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

- API base: `http://127.0.0.1:8000`
- OpenAPI docs: `http://127.0.0.1:8000/docs`
- Health check: `GET /health`

### Optional environment variables

| Variable | Description |
|----------|-------------|
| `CARDGRADING_CORS_ORIGINS` | JSON array of allowed browser origins, e.g. `["http://localhost:5173","https://app.example.com"]` |
| `CARDGRADING_WARP_HEIGHT` | Height in pixels of the normalized card crop (default `1120`) |
| `CARDGRADING_API_TITLE` | FastAPI `title` string (default `Card Grading AI`) |

If unset, CORS allows `http://localhost:5173` and `http://127.0.0.1:5173`. You can also change defaults in `backend/app/core/config.py`.

## Run the frontend

From the `frontend` directory:

```bash
npm install
npm run dev
```

Open the URL Vite prints (usually `http://localhost:5173`). The UI posts images to the backend at `http://127.0.0.1:8000` by default.

### Frontend environment

Set `VITE_API_BASE` if the API is not on `http://127.0.0.1:8000` (for example in a `.env` file in `frontend/`):

```env
VITE_API_BASE=https://your-api.example.com
```

## API

### `POST /analyze-card`

- **Body:** `multipart/form-data` with one field `file` (image)
- **Response:** JSON, for example:

```json
{
  "centering": {
    "left_right": "55/45",
    "top_bottom": "60/40"
  },
  "estimated_grades": {
    "PSA": 8.0,
    "BGS": 8.5,
    "CGC": 8.0
  },
  "warp_width": 784,
  "warp_height": 1120,
  "detection_confidence": "medium"
}
```

`detection_confidence` is a coarse label (`low` / `medium` / `high`) for the contour-based card detection step.

## Production build (frontend)

```bash
cd frontend
npm run build
```

Static output is written to `frontend/dist/`; serve it behind any static host or reverse proxy and point `VITE_API_BASE` at your deployed API at build time.

## Disclaimer

Grade estimates are **heuristic** and based only on automated centering signals. They are not affiliated with PSA, Beckett, CGC, or any grading company.
