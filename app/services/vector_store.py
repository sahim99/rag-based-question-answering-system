import os
import pickle
import faiss
import numpy as np
from typing import List, Dict, Any
from app.core.logging import setup_logging
from app.core.config import settings

logger = setup_logging()

# Use configurable data directory for Fly.io volume support
INDEX_DIR = os.path.join(settings.DATA_DIR, "faiss_index")
INDEX_FILE = os.path.join(INDEX_DIR, "index.faiss")
METADATA_FILE = os.path.join(INDEX_DIR, "metadata.pkl")

class VectorStore:
    def __init__(self):
        self.index = None
        self.metadata = {}  # Map vector_id (int) -> metadata (dict)
        self.dimension = None
        
        # Ensure directory exists
        os.makedirs(INDEX_DIR, exist_ok=True)
        
    def load_index(self):
        """
        Loads the FAISS index and metadata from disk if they exist.
        Otherwise initializes a new state.
        """
        if os.path.exists(INDEX_FILE) and os.path.exists(METADATA_FILE):
            try:
                self.index = faiss.read_index(INDEX_FILE)
                with open(METADATA_FILE, "rb") as f:
                    self.metadata = pickle.load(f)
                
                self.dimension = self.index.d
                logger.info(f"FAISS index loaded. Vectors: {self.index.ntotal}")
            except Exception as e:
                logger.error(f"Error loading FAISS index: {e}")
                self._initialize_empty_index()
        else:
            logger.info("No existing FAISS index found. Starting fresh.")
            self._initialize_empty_index()

    def _initialize_empty_index(self):
        self.index = None # Will be initialized on first add
        self.metadata = {}
        self.dimension = None

    def save_index(self):
        """
        Persists the current index and metadata to disk.
        """
        if self.index is None:
            logger.warning("Attempted to save empty index. Skipping.")
            return

        try:
            faiss.write_index(self.index, INDEX_FILE)
            with open(METADATA_FILE, "wb") as f:
                pickle.dump(self.metadata, f)
            logger.info(f"FAISS index saved to disk. Total vectors: {self.index.ntotal}")
        except Exception as e:
            logger.error(f"Error saving FAISS index: {e}")

    def add_embeddings(self, embeddings: List[List[float]], metadatas: List[Dict[str, Any]]):
        """
        Adds embeddings to the FAISS index and stores associated metadata.
        """
        if not embeddings:
            return

        count = len(embeddings)
        dim = len(embeddings[0])
        
        # Initialize index if first time
        if self.index is None:
            self.dimension = dim
            self.index = faiss.IndexFlatL2(dim)
            logger.info(f"Initialized new FAISS index with dimension: {dim}")

        if dim != self.dimension:
            logger.error(f"Embedding dimension mismatch. Expected {self.dimension}, got {dim}")
            return

        # Convert to numpy array
        vectors = np.array(embeddings).astype('float32')
        
        # Add to FAISS
        start_id = self.index.ntotal
        self.index.add(vectors)
        
        # Store metadata
        for i, meta in enumerate(metadatas):
            vector_id = start_id + i
            self.metadata[vector_id] = meta
            
        logger.info(f"Added {count} vectors to FAISS. New total: {self.index.ntotal}")
        
    def similarity_search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict]:
        """Performs vector similarity search and returns top-k results."""
        if self.index is None or self.index.ntotal == 0:
            logger.warning("Index is empty. Cannot search.")
            return []

        vector = np.array([query_embedding]).astype('float32')
        distances, indices = self.index.search(vector, top_k)
        
        results = []
        for j, i in enumerate(indices[0]):
            if i == -1: continue # No match
            if i in self.metadata:
                results.append({
                     "score": float(distances[0][j]),
                     "metadata": self.metadata[i]
                })
        
        logger.info(f"Internal Similarity Search completed. Found {len(results)} matches.")
        return results

# Global instance
vector_store = VectorStore()
