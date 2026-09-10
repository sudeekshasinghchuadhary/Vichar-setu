from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from ..database import get_db
from ..models import Scheme
from ..schemas import MatchRequest, MatchResponse, MatchItem
from ..intelligence.adapter import match_schemes

router = APIRouter(prefix="/match", tags=["Matching"])

def scheme_dict(s: Scheme) -> dict:
    return {"id": s.id, "scheme_name": s.scheme_name, "description": s.description,
            "min_age": s.min_age, "max_age": s.max_age, "max_income": s.max_income,
            "occupation": s.occupation, "education": s.education, "category": s.category,
            "state": s.state, "benefit": s.benefit}

@router.post("", response_model=MatchResponse)
def match(payload: MatchRequest, db: Session = Depends(get_db)):
    schemes = db.query(Scheme).order_by(Scheme.id).all()
    if not schemes:
        raise HTTPException(status_code=404, detail="No schemes found")
    raw = match_schemes(payload.profile.model_dump(), [scheme_dict(s) for s in schemes])
    results = sorted(raw, key=lambda x: x.get("match_score", 0), reverse=True)[:payload.top_k]
    for i, item in enumerate(results, start=1):
        item["rank"] = i
    return MatchResponse(results=[MatchItem(**item) for item in results])
