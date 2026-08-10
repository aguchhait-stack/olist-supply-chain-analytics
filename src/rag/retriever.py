import logging

from src.rag.embeddings import create_query_embedding


logger = logging.getLogger(__name__)


def retrieve_knowledge(
    question: str,
    collection,
    top_k: int = 3
) -> list[dict]:
    """
    Retrieve the most relevant knowledge documents
    for a user question.
    """

    logger.info(
        "Retrieving knowledge for question: %s",
        question
    )

    # Convert question into an embedding
    question_vector = create_query_embedding(question)

    # Search ChromaDB
    results = collection.query(
        query_embeddings=question_vector.tolist(),
        n_results=top_k
    )

    # Extract results
    ids = results["ids"][0]
    documents = results["documents"][0]
    distances = results["distances"][0]

    # Create a clean result list
    retrieved_documents = []

    for document_id, document, distance in zip(
        ids,
        documents,
        distances
    ):
        retrieved_documents.append(
            {
                "id": document_id,
                "document": document,
                "distance": distance
            }
        )

    logger.info(
        "Retrieved %d documents",
        len(retrieved_documents)
    )

    return retrieved_documents