from sentence_transformers import SentenceTransformer
import logging
logger = logging.getLogger(__name__)
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

model = SentenceTransformer(EMBEDDING_MODEL_NAME)
logger.info(f"Model loaded: {EMBEDDING_MODEL_NAME}")

def create_embeddings(documents: list[str]):
    """
    Convert documents into embeddings.
    """
    logger.info(f"Embedding {len(documents)} documents...")
    return model.encode(documents)


def create_query_embedding(question: str):
    """
    Convert a user question into an embedding.
    """
    return model.encode([question])