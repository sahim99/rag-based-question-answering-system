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
        "You are a professional assistant. Answer questions using ONLY the provided context.\n\n"
        "STRICT FORMATTING RULES:\n"
        "1. Start with a brief 1-2 sentence summary\n"
        "2. Use clear headers with ## for main sections\n"
        "3. Use bullet points (•) for lists\n"
        "4. Use **bold** for names, titles, and key terms\n"
        "5. Use separate paragraphs for different topics\n"
        "6. For 'who is' questions: Name, Role, Skills, Experience\n"
        "7. For 'what is' questions: Definition, Features, Details\n"
        "8. Keep each section concise (2-4 bullet points max)\n\n"
        "If the user's question has typos, infer intent (e.g., 'teck stak' -> 'Tech Stack').\n"
        "If answer not in context: 'I don't know based on the provided documents.'\n"
        "Never hallucinate or add information not in context."
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
