import httpx
from typing import List
from app.core.config import settings
from app.core.logging import setup_logging

logger = setup_logging()

async def generate_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Generates embeddings for a list of texts using Jina AI's API.
    """
    if not texts:
        return []
        
    url = "https://api.jina.ai/v1/embeddings"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {settings.JINA_API_KEY}"
    }
    data = {
        "input": texts,
        "model": "jina-embeddings-v3" 
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(url, headers=headers, json=data)
            response.raise_for_status()
            result = response.json()
            # Jina returns { "data": [ { "embedding": [...] } ] }
            embeddings = [item["embedding"] for item in result["data"]]
            return embeddings
    except Exception as e:
        logger.error(f"Error generating Jina embeddings: {e}")
        raise e
