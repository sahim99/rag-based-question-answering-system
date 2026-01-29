from fastapi import APIRouter, UploadFile, File, BackgroundTasks, HTTPException
from app.api.schemas import QueryRequest, QueryResponse, UploadResponse
from app.core.logging import setup_logging
from app.services.ingestion import save_upload_file, process_document

logger = setup_logging()

router = APIRouter()

@router.get("/health")
async def health_check():
    logger.info("Health check endpoint hit")
    return {"status": "ok"}

@router.post("/upload", response_model=UploadResponse)
async def upload_document(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    logger.info(f"Received upload request for file: {file.filename}")
    
    # Validate file extension
    if not (file.filename.endswith(".pdf") or file.filename.endswith(".txt")):
        raise HTTPException(status_code=400, detail="Invalid file type. Only PDF and TXT are allowed.")

    # Save file
    file_path = await save_upload_file(file)
    
    # Trigger background task
    background_tasks.add_task(process_document, file_path, file.filename)
    
    return {"message": "Upload received. Ingestion started in background."}

from app.services.retrieval import retrieve_context
from app.services.llm import generate_answer

@router.post("/query", response_model=QueryResponse)
async def query_document(request: QueryRequest):
    logger.info(f"Received query: {request.question}")
    
    # 1. Retrieve Context
    context_results = await retrieve_context(request.question)
    
    if not context_results:
        # Fallback if no context found or error
        logger.warning("No relevant context found.")
        return {
            "answer": "I don't know based on the provided documents (No relevant matches found).", 
            "sources": []
        }

    # 2. Generate Answer
    answer = await generate_answer(request.question, context_results)
    
    # 3. Format Response
    sources = []
    for res in context_results:
        meta = res.get('metadata', {})
        sources.append({
            "source_file": meta.get('source_file', 'unknown'),
            "chunk_id": meta.get('chunk_id', 'unknown')
        })
        
    return {"answer": answer, "sources": sources}
