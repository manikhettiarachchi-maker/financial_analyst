#from fastapi import FastAPI
#app = FastAPI()

#@app.get("/")
#def home():
    #return {"status": "ok"}
#@app.get("/")
#def home():
    #return {"status": "ok"}

# ------17/02/2026---
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging
import os

# Import your agent runner function
# IMPORTANT: this must NOT execute the agent at import time
from app import run_agent  # <-- make sure app.py defines run_agent(query: str) -> str

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Financial Analyst AI", version="1.0.0")


class AnalyzeRequest(BaseModel):
    query: str


@app.get("/")
def root():
    return {"status": "ok"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    try:
        if not req.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")

        # Call your agent
        result = run_agent(req.query)

        # Normalize response (some frameworks return dicts/objects)
        return {"query": req.query, "result": result}

    except HTTPException:
        raise
    except Exception as e:
        logging.exception("Error during /analyze")
        raise HTTPException(status_code=500, detail=str(e))
