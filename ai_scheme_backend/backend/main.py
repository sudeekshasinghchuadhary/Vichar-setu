from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .config import settings
from .routes import profile, schemes, eligibility, matching

app = FastAPI(title="AI Driven Scheme Matching API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(profile.router)
app.include_router(schemes.router)
app.include_router(eligibility.router)
app.include_router(matching.router)

@app.get("/")
def root():
    return {"message": "AI Driven Scheme Matching API is running"}

@app.get("/health")
def health():
    return {"status": "ok"}
