from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from ..database import get_db
from ..models import Scheme
from ..schemas import SchemeResponse

router = APIRouter(prefix="/schemes", tags=["Schemes"])

@router.get("", response_model=list[SchemeResponse])
def get_schemes(db: Session = Depends(get_db)):
    return db.query(Scheme).order_by(Scheme.id).all()

@router.get("/{scheme_id}", response_model=SchemeResponse)
def get_scheme(scheme_id: int, db: Session = Depends(get_db)):
    scheme = db.get(Scheme, scheme_id)
    if not scheme:
        raise HTTPException(status_code=404, detail="Scheme not found")
    return scheme
