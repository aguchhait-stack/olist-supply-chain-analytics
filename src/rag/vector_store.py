import chromadb
import logging

logger = logging.getLogger(__name__)

def create_vector_store(documents,document_vectors):
    """
    Storing embeddings in a ChromaDB collection.
    """
    # Create a persistant client
    client = chromadb.PersistentClient(path="vectorstore/chroma")
    # Create a database
    collection = client.get_or_create_collection(name="business_knowledge",
                                                configuration={"hnsw": { "space": "cosine"}})
    collection.add(
        ids=[str(i) for i in range(len(documents))],
        documents=documents,
        embeddings=document_vectors.tolist())
    logger.info("Vector store created successfully with %d documents",collection.count())
    logger.info(f"Configuration:{collection.configuration}")
    return collection

def get_collection():
    """Get the existing collection."""
    client = chromadb.PersistentClient(path="vectorstore/chroma")
    return client.get_collection("business_knowledge")