import chromadb
import logging

logger = logging.getLogger(__name__)

def create_vector_store(
    documents,
    document_vectors
):
    """
    Store documents and embeddings in a persistent ChromaDB collection.
    """

    logger.info("Creating ChromaDB vector store")

    client = chromadb.PersistentClient(
        path="vectorstore/chroma"
    )

    collection = client.get_or_create_collection(
        name="business_knowledge"
    )

    logger.info(
        "Adding %d documents to vector store",
        len(documents)
    )

    collection.add(
        ids=[str(i) for i in range(len(documents))],
        documents=documents,
        embeddings=document_vectors.tolist()
    )

    logger.info(
        "Vector store created successfully with %d documents",
        collection.count()
    )

    return collection