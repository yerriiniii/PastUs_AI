# ai/app.py
from fastapi import FastAPI
from ai.routers.tagging import router as tagging_router

app = FastAPI()

app.include_router(tagging_router)


@app.get("/health")
def health_check():
    return {"status": "ok"}
