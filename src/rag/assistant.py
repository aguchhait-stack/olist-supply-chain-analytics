import logging

from transformers import pipeline

from src.rag.retriever import retrieve_knowledge


logger = logging.getLogger(__name__)

HF_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


# ============================================================
# Hugging Face LLM
# ============================================================

logger.info(
    "Loading Hugging Face model: %s",
    HF_MODEL_NAME
)

generator = pipeline(
    task="text-generation",
    model=HF_MODEL_NAME
)

logger.info(
    "Hugging Face model loaded successfully."
)


# ============================================================
# Build context
# ============================================================

def build_context(
    question: str,
    collection,
    top_k: int = 3
) -> str:
    """
    Retrieve relevant documents and build the context.
    """

    results = retrieve_knowledge(
        question=question,
        collection=collection,
        top_k=top_k
    )

    if not results:
        logger.warning(
            "No relevant documents found."
        )

        return ""

    context = "\n\n".join(
        result["document"]
        for result in results
    )

    logger.info(
        "Built context from %d documents",
        len(results)
    )

    return context


# ============================================================
# Build prompt
# ============================================================

def build_prompt(
    question: str,
    context: str
) -> str:
    """
    Build the prompt sent to the LLM.
    """

    return f"""
You are a business analytics assistant.

Answer the user's question using the retrieved business
knowledge provided below.

Instructions:
- Use the most relevant information needed to answer the question.
- Do not assume that every retrieved document is relevant.
- Do not mention or summarize every retrieved document unless necessary.
- Use only facts supported by the retrieved knowledge.
- Do not invent statistics, calculations, states, dates, trends, or causes.
- Do not perform mathematical calculations unless the required values
  are explicitly provided.
- Do not treat correlation as causation.
- If the retrieved knowledge does not contain enough information,
  clearly say that the available knowledge is insufficient.
- Give a concise and direct business answer.
- Do not repeat the same point.
- Do not add greetings, conclusions, or offers for further assistance.


Business knowledge:
{context}

User question:
{question}

Provide a concise, evidence-based business answer.
""".strip()


# ============================================================
# Generate answer
# ============================================================

def generate_answer(
    question: str,
    context: str
) -> str:
    """
    Generate the final answer using
    the local Hugging Face LLM.
    """

    prompt = build_prompt(
        question=question,
        context=context
    )

    logger.info(
        "Generating answer using: %s",
        HF_MODEL_NAME
    )

    result = generator(
        prompt,
        max_new_tokens=100,
        do_sample=False
    )

    generated_text = result[0]["generated_text"]

    # Remove the original prompt from the generated output.
    answer = generated_text[len(prompt):].strip()

    logger.info(
        "Successfully generated answer."
    )

    return answer


# ============================================================
# Complete RAG pipeline
# ============================================================

def ask_assistant(
    question: str,
    collection,
    top_k: int = 3
) -> str:
    """
    Run the complete RAG pipeline.

    Question
        ↓
    Retrieval
        ↓
    Context
        ↓
    Prompt
        ↓
    Hugging Face LLM
        ↓
    Answer
    """

    logger.info(
        "Starting RAG pipeline."
    )

    context = build_context(
        question=question,
        collection=collection,
        top_k=top_k
    )

    if not context:
        return (
            "I could not find relevant business "
            "knowledge to answer this question."
        )

    answer = generate_answer(
        question=question,
        context=context
    )

    logger.info(
        "RAG pipeline completed."
    )

    return answer