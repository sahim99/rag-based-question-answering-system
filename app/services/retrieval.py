from typing import List, Dict
from app.services.vector_store import vector_store
from app.services.embeddings import generate_embeddings
from app.core.logging import setup_logging

logger = setup_logging()

async def retrieve_context(query: str, top_k: int = 5) -> List[Dict]:
    """
    Retrieves relevant context for a given query.
    1. Generate embedding for query.
    2. Search vector store.
    3. Return list of metadata (with text).
    """
    try:
        # Generate embedding
        # generate_embeddings returns List[List[float]], we take the first one
        embeddings = await generate_embeddings([query])
        if not embeddings:
            logger.warning("Failed to generate embedding for query.")
            return []
            
        query_embedding = embeddings[0]
        
        # Search FAISS
        results = vector_store.similarity_search(query_embedding, top_k=top_k)
        
        logger.info(f"Retrieved {len(results)} chunks for query.")
        return results
    except Exception as e:
        logger.error(f"Error during retrieval: {e}")
        return []
