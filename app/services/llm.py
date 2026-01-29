import time
from groq import AsyncGroq
from app.core.config import settings
from app.core.logging import setup_logging

logger = setup_logging()

# Initialize Groq client
client = AsyncGroq(api_key=settings.GROQ_API_KEY)

async def generate_answer(query: str, context_chunks: list) -> str:
    """
    Generates an answer using Groq LLM based on the provided context.
    1. Construct Prompt.
    2. Call LLM.
    3. Return answer.
    """
    start_time = time.time()
    
    # 1. Construct Context String
    context_text = ""
    for chunk in context_chunks:
        # access metadata from structure returned by vector_store.similarity_search
        # result item structure: {'score': float, 'metadata': {...}}
        meta = chunk.get('metadata', {})
        text = meta.get('text', '')
        context_text += f"<chunk source='{meta.get('source_file')}' id='{meta.get('chunk_id')}'>\n{text}\n</chunk>\n\n"

    # 2. Construct Prompt
    system_prompt = (
        "You are a helpful assistant answering questions using ONLY the provided context.\n"
        "If the user's question contains typos, infer the intent (e.g., 'tack stak' -> 'Tech Stack').\n\n"
        "FORMATTING RULES:\n"
        "- Structure your answer clearly with sections if appropriate\n"
        "- Use bullet points for lists\n"
        "- Use **bold** for key terms\n"
        "- Keep answers concise but comprehensive\n"
        "- If the answer has multiple parts, organize them logically\n\n"
        "If the answer is not in the context, say 'I don't know based on the provided documents.'\n"
        "Do not hallucinate or add information not in the context."
    )
    
    user_prompt = f"Context:\n{context_text}\n\nQuestion:\n{query}"

    try:
        # 3. Call Groq
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
            model="llama-3.3-70b-versatile", # Updated to latest stable model
            temperature=0, # Deterministic
        )
        
        answer = chat_completion.choices[0].message.content
        
        latency = (time.time() - start_time) * 1000
        logger.info(f"LLM Response generated in {latency:.2f}ms. Model: llama3-8b-8192")
        
        return answer

    except Exception as e:
        logger.error(f"Error calling LLM: {e}")
        return "Sorry, I encountered an error while processing your request."
