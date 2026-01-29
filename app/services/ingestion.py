import os
import shutil
from fastapi import UploadFile, HTTPException
from pypdf import PdfReader
from app.services.chunking import chunk_text
from app.services.embeddings import generate_embeddings
from app.services.vector_store import vector_store
from app.core.logging import setup_logging
from app.core.config import settings

logger = setup_logging()

# Use configurable data directory for Fly.io volume support
UPLOAD_DIR = os.path.join(settings.DATA_DIR, "uploads")

# Ensure upload directory exists
os.makedirs(UPLOAD_DIR, exist_ok=True)

async def process_document(file_path: str, filename: str):
    """
    Background task to process the document: read, chunk, embed, log.
    """
    logger.info(f"Starting ingestion for file: {filename}")
    
    data = ""
    try:
        if filename.endswith(".pdf"):
            reader = PdfReader(file_path)
            # Extract text from all pages
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    data += text + "\n"
        elif filename.endswith(".txt"):
            with open(file_path, "r", encoding="utf-8") as f:
                data = f.read()
        
        logger.info(f"Text extraction complete for {filename}. Length: {len(data)} chars")
        
        # Chunking
        chunks = chunk_text(data, filename)
        logger.info(f"Created {len(chunks)} chunks for {filename}")
        
        if not chunks:
            logger.warning(f"No text extracted from {filename}. Skipping embeddings.")
            return

        # Prepare texts for embedding
        chunk_texts = [chunk["text"] for chunk in chunks]
        
        # Generate Embeddings
        # Note: In a real production scenario, we might want to batch these if there are too many
        embeddings = await generate_embeddings(chunk_texts)
        logger.info(f"Generated {len(embeddings)} embeddings for {filename}")
        
        # Prepare metadata
        metadatas = []
        for chunk in chunks:
            metadatas.append({
                "text": chunk["text"],
                "source_file": chunk["source_file"],
                "chunk_id": chunk["chunk_id"]
            })
            
        # Add to Vector Store and Persist
        vector_store.add_embeddings(embeddings, metadatas)
        vector_store.save_index()
        
        logger.info(f"Ingestion pipeline completed successfully for {filename}")

    except Exception as e:
        logger.error(f"Failed to ingest {filename}: {str(e)}")
        # In production, we'd update a job status in DB here

async def save_upload_file(upload_file: UploadFile) -> str:
    """
    Saves the uploaded file to disk and returns the path.
    """
    try:
        file_path = os.path.join(UPLOAD_DIR, upload_file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(upload_file.file, buffer)
        return file_path
    except Exception as e:
        logger.error(f"Error saving file {upload_file.filename}: {e}")
        raise HTTPException(status_code=500, detail="Could not save file")
