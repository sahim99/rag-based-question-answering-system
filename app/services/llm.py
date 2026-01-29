import time
from typing import List, Dict, Any
from groq import AsyncGroq
from app.core.config import settings
from app.core.logging import setup_logging

logger = setup_logging()

# Constants
MODEL_NAME = "llama-3.3-70b-versatile"
TEMPERATURE = 0  # Deterministic output

# Initialize Groq client
client = AsyncGroq(api_key=settings.GROQ_API_KEY)

async def generate_answer(query: str, context_chunks: List[Dict[str, Any]]) -> str:
    """
    Generates a deterministic, grounded answer using Groq LLM based on the provided context.
    
    This function:
    1. Constructs a safe context string from retrieved chunks.
    2. Builds a strict system prompt to prevent hallucination.
    3. Calls the LLM with deterministic settings.
    4. Logs latency and model usage for observability.
    
    Args:
        query (str): The user's question.
        context_chunks (List[Dict[str, Any]]): List of retrieved chunks with metadata.
        
    Returns:
        str: The generated answer or a graceful error message.
    """
    start_time = time.time()
    
    # 1. Construct Robust Context String
    context_text = ""
    for chunk in context_chunks:
        # Safely access metadata with defaults to prevent crashes
        meta = chunk.get('metadata', {})
        text = meta.get('text', '').strip()
        source = meta.get('source_file', 'unknown')
        chunk_id = meta.get('chunk_id', 'unknown')
        
        if text:
            context_text += f"<chunk source='{source}' id='{chunk_id}'>\n{text}\n</chunk>\n\n"

    # Handle empty context case gracefully
    if not context_text:
        logger.warning("No context provided for query.")
        context_text = "No relevant context found."

    # 2. Construct Strict System Prompt
    system_prompt = (
        "You are a professional assistant. Answer questions using ONLY the provided context.\n\n"
        "STRICT FORMATTING RULES:\n"
        "1. Start with a direct, high-level summary (1-2 sentences).\n"
        "2. Use clear headers with '##' to organize main topics.\n"
        "3. Use bullet points (•) for all lists and key details.\n"
        "4. Use **bold** for important entities (names, tools, dates, metrics).\n"
        "5. Keep paragraphs short and readable.\n"
        "6. If the context contains code, format it properly.\n"
        "7. Ensure the answer flows logically and is easy to scan.\n\n"
        "SAFETY & GROUNDING RULES:\n"
        "- Infer user intent if there are typos (e.g., 'teck stak' -> 'Tech Stack').\n"
        "- STICK STRICTLY TO THE CONTEXT. Do not use outside knowledge.\n"
        "- If the answer is partially available, provide what is there and mention what is missing.\n"
        "- If the answer is NOT in the context, say exactly: 'I don't know based on the provided documents.'\n"
        "- Do not fabricate information."
    )
    
    user_prompt = f"Context:\n{context_text}\n\nQuestion:\n{query}"

    try:
        # 3. Call Groq LLM
        chat_completion = await client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                }
            ],
            model=MODEL_NAME,
            temperature=TEMPERATURE,
        )
        
        answer = chat_completion.choices[0].message.content
        
        # 4. Log Metrics
        latency_ms = (time.time() - start_time) * 1000
        logger.info(f"LLM Response generated in {latency_ms:.2f}ms. Model: {MODEL_NAME}")
        
        return answer

    except Exception as e:
        # Log the full error for debugging but return a safe message to the user
        logger.error(f"Error calling LLM: {str(e)}")
        return "I apologize, but I encountered an error while processing your request. Please try again later."
