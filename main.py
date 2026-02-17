from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app import run_agent  # run_agent(query: str) -> dict | str

app = FastAPI(title="Financial Analyst API")


class AnalyzeRequest(BaseModel):
    query: str


@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    try:
        # IMPORTANT: pass only the query string (not the whole dict)
        result = run_agent(req.query)

        # If your run_agent returns a plain string, wrap it.
        if isinstance(result, str):
            return {"result": result}

        # If it returns a dict already, return it directly.
        return result

    except Exception as e:
        # Avoid crashing the worker; return a 500 instead.
        raise HTTPException(status_code=500, detail=str(e))
