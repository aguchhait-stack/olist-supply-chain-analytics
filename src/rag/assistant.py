import logging

from src.rag.retriever import retrieve_knowledge


logger = logging.getLogger(__name__)


def build_context(
    question: str,
    collection,
    top_k: int = 3
) -> str:
    """
    Retrieve relevant documents and combine them
    into context for the assistant.
    """

    results = retrieve_knowledge(
        question,
        collection,
        top_k
    )

    context = "\n\n".join(
        result["document"]
        for result in results
    )

    logger.info(
        "Built context from %d retrieved documents",
        len(results)
    )

    return context