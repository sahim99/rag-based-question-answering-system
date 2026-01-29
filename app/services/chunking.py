from typing import List, Dict

def chunk_text(text: str, source_file: str, chunk_size: int = 1000, overlap: int = 200) -> List[Dict]:
    """
    Splits text into chunks of `chunk_size` characters with `overlap`.
    """
    if not text:
        return []
        
    chunks = []
    text_len = len(text)
    start = 0
    
    while start < text_len:
        end = start + chunk_size
        chunk_text = text[start:end]
        
        chunks.append({
            "text": chunk_text,
            "chunk_id": f"{source_file}_chunk_{start}",
            "source_file": source_file
        })
        
        # If we reached the end, stop
        if end >= text_len:
            break
            
        # Move start forward by (chunk_size - overlap)
        start += (chunk_size - overlap)
        
    return chunks
