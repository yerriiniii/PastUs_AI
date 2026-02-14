from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

load_dotenv()

from ai.routers.recommendation.recommendation import router as recommendation_router
from ai.routers.tagging.tagging import router as tagging_router

app = FastAPI()
app.include_router(tagging_router)
app.include_router(recommendation_router)

@app.exception_handler(Exception)
def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            "detail": "오류가 발생했습니다.",
            "type": type(exc).__name__,
        },
    )


@app.get("/health")
def health_check():
    return {"status": "ok"}
