from sentence_transformers import SentenceTransformer


EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

model = SentenceTransformer(
    EMBEDDING_MODEL_NAME
)


def create_embeddings(documents: list[str]):
    """
    Convert documents into embeddings.
    """
    return model.encode(documents)


def create_query_embedding(question: str):
    """
    Convert a user question into an embedding.
    """
    return model.encode([question])