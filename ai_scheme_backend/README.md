# AI Driven Scheme Matching — Raj + Anusha Backend

This package contains **only the FastAPI/backend work** assigned to Raj + Anusha:

- FastAPI project structure
- Profile API
- Scheme API
- Eligibility API
- Matching API
- Pydantic schemas
- PostgreSQL connection using the existing DB schema
- Intelligence-engine integration adapter
- Basic errors
- CORS for React/Vite

## Important scope boundary

Do **not** modify the PostgreSQL schema or add scheme data here. Database/schema/data work belongs to Kushagra. Core AI/rule/matching logic belongs to Ananya.

The current backend maps API names to the existing DB columns:

- `social_category` → `users.category`
- `annual_family_income` → `users.annual_income`
- `education_level` → `users.education`

`purpose`, `project_type`, and `estimated_project_cost` are accepted by the API but are not persisted because the existing `users` table does not contain those columns. They remain available to the intelligence engine during `/eligibility` and `/match` requests.

## 1. Install

```bash
python -m venv venv
```

Windows:
```bash
venv\Scripts\activate
```

macOS/Linux:
```bash
source venv/bin/activate
```

```bash
pip install -r requirements.txt
```

## 2. Configure PostgreSQL

Copy `.env.example` to `.env` and put the actual PostgreSQL credentials/database name:

```env
DATABASE_URL=postgresql+psycopg2://postgres:YOUR_PASSWORD@localhost:5432/YOUR_DATABASE
CORS_ORIGINS=http://localhost:5173,http://127.0.0.1:5173
ENGINE_MODE=mock
```

The database must already contain the tables from the team's PostgreSQL database/dump.

## 3. Run

From the project root (`ai_scheme_backend`):

```bash
uvicorn backend.main:app --reload
```

Open:

- `http://127.0.0.1:8000/docs`
- `http://127.0.0.1:8000/health`

## 4. API endpoints

### POST `/profile`
Creates a profile in the existing `users` table.

Example body:
```json
{
  "name": "Raj",
  "age": 24,
  "gender": "Male",
  "social_category": "SC",
  "state": "Uttar Pradesh",
  "district": "Lucknow",
  "occupation": "Small Business",
  "annual_family_income": 300000,
  "purpose": "Start a new business",
  "project_type": "Manufacturing",
  "estimated_project_cost": 500000,
  "education_level": "Graduate"
}
```

### GET `/schemes`
Returns schemes from the existing `schemes` table.

### GET `/schemes/{scheme_id}`
Returns one scheme.

### POST `/eligibility`
Checks one profile against one scheme through the intelligence adapter.

Example:
```json
{
  "scheme_id": 2,
  "profile": {
    "age": 24,
    "social_category": "SC",
    "state": "Uttar Pradesh",
    "district": "Lucknow",
    "occupation": "Small Business",
    "annual_family_income": 300000,
    "purpose": "Start a business",
    "project_type": "Manufacturing",
    "estimated_project_cost": 500000,
    "education_level": "Graduate"
  }
}
```

### POST `/match`
Fetches schemes from PostgreSQL and sends them to the intelligence engine for ranking.

Example:
```json
{
  "top_k": 5,
  "profile": {
    "age": 24,
    "social_category": "SC",
    "state": "Uttar Pradesh",
    "district": "Lucknow",
    "occupation": "Small Business",
    "annual_family_income": 300000,
    "purpose": "Start a business",
    "project_type": "Manufacturing",
    "estimated_project_cost": 500000,
    "education_level": "Graduate"
  }
}
```

## 5. Connecting Ananya's intelligence engine

Ananya should provide a Python module named `intelligence_engine.py` that exposes:

```python
def check_eligibility(profile: dict, scheme: dict) -> dict:
    ...


def match_schemes(profile: dict, schemes: list[dict]) -> list[dict]:
    ...
```

Then place `intelligence_engine.py` in the project root (next to this README) or make it importable in the Python environment, and change:

```env
ENGINE_MODE=real
```

Expected eligibility result:
```json
{
  "eligible": true,
  "reasons": ["..."],
  "status": "VERIFIED"
}
```

Expected matching item:
```json
{
  "scheme_id": 2,
  "match_score": 87.5,
  "rank": 1,
  "reasons": ["..."],
  "eligible": true
}
```

The `mock` mode is only for testing the FastAPI integration before Ananya's engine is merged. It is **not** the final AI engine.

## 6. React integration

Frontend calls can use:

```js
const API = "http://127.0.0.1:8000";

const response = await fetch(`${API}/match`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ profile, top_k: 10 })
});

const data = await response.json();
```

CORS is already enabled for the Vite development URLs.

## Folder structure

```text
ai_scheme_backend/
├── backend/
│   ├── __init__.py
│   ├── main.py
│   ├── config.py
│   ├── database.py
│   ├── models.py
│   ├── schemas.py
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── profile.py
│   │   ├── schemes.py
│   │   ├── eligibility.py
│   │   └── matching.py
│   └── intelligence/
│       ├── __init__.py
│       ├── adapter.py
│       └── engine_contract.py
├── .env.example
├── requirements.txt
└── README.md
```
