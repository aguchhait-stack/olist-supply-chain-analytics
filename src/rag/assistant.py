import logging
from google import genai
from dotenv import load_dotenv
import os
from src.rag.retriever import retrieve_knowledge

logger = logging.getLogger(__name__)

load_dotenv()
GEMINI_MODEL = "gemini-3.5-flash"

def ask_assistant(question: str, collection, top_k: int = 2) -> str:
    """RAG pipeline — retrieve context and generate answer."""
    try:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return "Please set up my API key." 
        client = genai.Client(api_key=api_key)

        # 1. Retrieve
        results = retrieve_knowledge(question, collection, top_k)
        if not results:
            logger.warning("No relevant documents found.")
            return "I could not find relevant knowledge to answer this question."
        
        context = "\n\n".join(r["document"] for r in results)
        
        # 2. Prompt
        prompt = f"""
    Answer the user's question using only the retrieved business knowledge below.
    Do not add unsupported information.
    Always respond in English.

    Retrieved knowledge:
    {context}

    Question:
    {question}

    Answer:
    """
        
        # 3. Generate
        response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
        logger.info("RAG pipeline completed.")
        return response.text.strip()
    except Exception as e:
        logger.error(f"RAG pipeline failed: {e}")