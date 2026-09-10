from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from ..database import get_db
from ..models import User
from ..schemas import ProfileCreate, ProfileResponse

router = APIRouter(prefix="/profile", tags=["Profile"])

@router.post("", response_model=ProfileResponse, status_code=201)
def create_profile(payload: ProfileCreate, db: Session = Depends(get_db)):
    user = User(
        name=payload.name, age=payload.age, gender=payload.gender,
        category=payload.social_category, state=payload.state, district=payload.district,
        annual_income=payload.annual_family_income, occupation=payload.occupation,
        education=payload.education_level, created_at=datetime.utcnow(),
    )
    db.add(user); db.commit(); db.refresh(user)
    return ProfileResponse(
        id=user.id, name=user.name, age=user.age, gender=user.gender,
        social_category=user.category, state=user.state, district=user.district,
        occupation=user.occupation, annual_family_income=user.annual_income,
        education_level=user.education,
    )
