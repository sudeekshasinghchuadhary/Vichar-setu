from datetime import datetime
from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship
from .database import Base

class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    age: Mapped[int | None] = mapped_column(Integer, nullable=True)
    gender: Mapped[str | None] = mapped_column(String(50), nullable=True)
    state: Mapped[str | None] = mapped_column(String(100), nullable=True)
    district: Mapped[str | None] = mapped_column(String(100), nullable=True)
    category: Mapped[str | None] = mapped_column(String(100), nullable=True)
    annual_income: Mapped[float | None] = mapped_column(Float, nullable=True)
    occupation: Mapped[str | None] = mapped_column(String(255), nullable=True)
    education: Mapped[str | None] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

class Scheme(Base):
    __tablename__ = "schemes"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    scheme_name: Mapped[str] = mapped_column(String(255))
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    ministry: Mapped[str | None] = mapped_column(String(255), nullable=True)
    state: Mapped[str | None] = mapped_column(String(100), nullable=True)
    category: Mapped[str | None] = mapped_column(String(100), nullable=True)
    min_age: Mapped[int | None] = mapped_column(Integer, nullable=True)
    max_age: Mapped[int | None] = mapped_column(Integer, nullable=True)
    max_income: Mapped[float | None] = mapped_column(Float, nullable=True)
    occupation: Mapped[str | None] = mapped_column(String(255), nullable=True)
    education: Mapped[str | None] = mapped_column(String(255), nullable=True)
    benefit: Mapped[str | None] = mapped_column(Text, nullable=True)
    documents_required: Mapped[str | None] = mapped_column(Text, nullable=True)
    application_link: Mapped[str | None] = mapped_column(Text, nullable=True)

class EligibilityCriterion(Base):
    __tablename__ = "eligibility_criteria"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    scheme_id: Mapped[int] = mapped_column(ForeignKey("schemes.id"))
    criterion_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    criterion_value: Mapped[str | None] = mapped_column(String(255), nullable=True)

class MatchingResult(Base):
    __tablename__ = "matching_results"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    scheme_id: Mapped[int] = mapped_column(ForeignKey("schemes.id"))
    eligibility_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    match_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
