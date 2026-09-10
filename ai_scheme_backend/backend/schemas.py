from pydantic import BaseModel, ConfigDict, Field
from typing import Any

class ProfileBase(BaseModel):
    name: str | None = None
    age: int | None = Field(default=None, ge=0, le=120)
    gender: str | None = None
    social_category: str | None = None
    state: str | None = None
    district: str | None = None
    occupation: str | None = None
    annual_family_income: float | None = Field(default=None, ge=0)
    purpose: str | None = None
    project_type: str | None = None
    estimated_project_cost: float | None = Field(default=None, ge=0)
    education_level: str | None = None

class ProfileCreate(ProfileBase):
    pass

class ProfileResponse(ProfileBase):
    id: int
    model_config = ConfigDict(from_attributes=True)

class SchemeResponse(BaseModel):
    id: int
    scheme_name: str
    description: str | None = None
    ministry: str | None = None
    state: str | None = None
    category: str | None = None
    min_age: int | None = None
    max_age: int | None = None
    max_income: float | None = None
    occupation: str | None = None
    education: str | None = None
    benefit: str | None = None
    documents_required: str | None = None
    application_link: str | None = None
    model_config = ConfigDict(from_attributes=True)

class EligibilityRequest(BaseModel):
    profile: ProfileBase
    scheme_id: int

class EligibilityResponse(BaseModel):
    scheme_id: int
    eligible: bool
    reasons: list[str]
    status: str = "VERIFIED"

class MatchRequest(BaseModel):
    profile: ProfileBase
    top_k: int = Field(default=10, ge=1, le=50)

class MatchItem(BaseModel):
    scheme_id: int
    match_score: float = Field(ge=0, le=100)
    rank: int
    reasons: list[str]
    eligible: bool | None = None

class MatchResponse(BaseModel):
    results: list[MatchItem]

class ErrorResponse(BaseModel):
    detail: str
    code: str | None = None
