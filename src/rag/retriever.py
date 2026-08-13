import logging

from src.rag.embeddings import create_query_embedding

logger = logging.getLogger(__name__)

def retrieve_knowledge(question: str,collection,top_k: int = 2, max_distance: float = 0.7) -> list:
    """
    Retrieve the most relevant knowledge documents
    for a user question.
    """

    # Convert question into an embedding
    question_vector = create_query_embedding(question)

    # Search ChromaDB
    results = collection.query(query_embeddings=question_vector,n_results=top_k)

    # Extract results
    ids = results["ids"][0]
    documents = results["documents"][0]
    distances = results["distances"][0]

    # Create a clean result list
    retrieved_documents = []

    for document_id, document, distance in zip(ids,documents,distances):
        if distance <= max_distance:
            retrieved_documents.append(
                {"id": document_id,
                "document": document,
                "distance": distance
                })

    logger.info(f"Retrieved documents: {len(retrieved_documents)}")

    return retrieved_documents