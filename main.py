from src.utils.logging import setup_logging
from src.data.ingest import engine,ingestion
from src.data.builder import build_logistics_dataframe
from src.analytics.rfm import build_customer_segmentation
from src.nlp.sentiment import build_sentiment_analysis
from src.rag.knowledge_base import build_knowledge_base
from src.rag.embeddings import create_embeddings
from src.rag.vector_store import create_vector_store
import pandas as pd
setup_logging()

def main():
    """
    Execute the complete analytics pipeline.
    """
    tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", engine)

    # Check if tables exist
    if len(tables) == 0:
        print("No tables found. Running ingestion...")
        engine.dispose()
        # Run full pipeline
        ingestion()
    else:
        print(f"{len(tables)} tables already loaded.")

    print("Starting logistics dataframe...")
    logistics_df = build_logistics_dataframe(engine)
    print("Logistics dataframe completed.")

    print("Starting customer segmentation...")
    logistics_df, rfm, X_train_scaled, X_test_scaled, kmeans = build_customer_segmentation(logistics_df)
    print("Customer segmentation completed.")

    print("Starting NLP...")
    nlp_df, word_frequency, X_matrix, vectorizer = build_sentiment_analysis(logistics_df)
    print("NLP completed.")

    print("Starting knowledge base...")
    documents = build_knowledge_base(logistics_df,contingency,nlp_df)
    print(f"Knowledge base created with {len(documents)} documents.")

    print("Starting embeddings...")
    document_vectors = create_embeddings(documents)
    print("Embeddings completed.")

    print("Starting ChromaDB...")
    collection = create_vector_store(documents,document_vectors)
    print(f"Vector store created with {collection.count()} documents.")

    print("Pipeline completed successfully.")

    return logistics_df, rfm, nlp_df, collection


if __name__ == "__main__":
    main()