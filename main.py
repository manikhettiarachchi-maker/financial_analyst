from fastapi import FastAPI
from app import run_agent   # change to your main function

app = FastAPI()

@app.get("/")
def home():
    return {"status": "AI Financial Analyst running"}

@app.post("/analyze")
def analyze(data: dict):
    result = run_agent(data)
    return result
