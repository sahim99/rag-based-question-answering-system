from fastapi import FastAPI
from app.core.config import settings
from app.core.logging import setup_logging
from app.api.routes import router

# Setup logging
logger = setup_logging()

app = FastAPI(title=settings.PROJECT_NAME, version=settings.VERSION)

from app.services.vector_store import vector_store

@app.on_event("startup")
async def startup_event():
    logger.info("Application starting up...")
    vector_store.load_index()

app.include_router(router, prefix="/api")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
