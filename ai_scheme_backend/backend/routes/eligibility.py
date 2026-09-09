from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from ..database import get_db
from ..models import Scheme
from ..schemas import EligibilityRequest, EligibilityResponse
from ..intelligence.adapter import check_eligibility

router = APIRouter(prefix="/eligibility", tags=["Eligibility"])

def scheme_dict(s: Scheme) -> dict:
    return {"id": s.id, "scheme_name": s.scheme_name, "description": s.description,
            "min_age": s.min_age, "max_age": s.max_age, "max_income": s.max_income,
            "occupation": s.occupation, "education": s.education, "category": s.category,
            "state": s.state, "benefit": s.benefit}

@router.post("", response_model=EligibilityResponse)
def eligibility(payload: EligibilityRequest, db: Session = Depends(get_db)):
    scheme = db.get(Scheme, payload.scheme_id)
    if not scheme:
        raise HTTPException(status_code=404, detail="Scheme not found")
    result = check_eligibility(payload.profile.model_dump(), scheme_dict(scheme))
    return EligibilityResponse(scheme_id=scheme.id, **result)
