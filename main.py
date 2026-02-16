from fastapi import FastAPI
from app import run_agent

app = FastAPI()

@app.get("/")
def home():
    return {"status": "ok"}

@app.post("/analyze")
def analyze(data: dict):
    return run_agent(data)
