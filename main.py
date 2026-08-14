from src.data.ingest import engine, ingestion
from src.data.logistics_data import build_master_dataframe, build_logistics_dataframe
from src.utils.validation import validate_dataframe, find_aggregation_columns
from src.analysis.rfm_segmentation import build_customer_segmentation
from src.analysis.logistics_analysis import analyze_sla_by_state
from src.analysis.nlp_analysis import build_sentiment_analysis
from src.models.sentiment_model import train_sentiment_model
from src.models.sentiment_comparison import compare_with_huggingface_baseline
from src.rag.knowledge_base import build_knowledge_base
from src.rag.embeddings import create_embeddings
from src.rag.vector_store import create_vector_store
from src.rag.assistant import ask_assistant
import pandas as pd
import logging
from src.utils.logger_utils import setup_logging 

setup_logging()
logger = logging.getLogger(__name__) 

def main():
    """Execute the complete analytics pipeline."""
    print("=" * 60)
    print("Olist Supply Chain Analytics")
    print("=" * 60)
    
    # Database
    tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", engine)
    if len(tables) == 0:
        logger.info("Ingesting data...")
        engine.dispose()
        ingestion()
    else:
        logger.info(f"Database ready: {len(tables)} tables")

    # Data preparation
    print("=" * 60)
    print("Master DataFrame")
    print("=" * 60)
    master_df = build_master_dataframe(engine)
    find_aggregation_columns(master_df)

    print("=" * 60)
    print("Logistics DataFrame")
    print("=" * 60)
    logistics_df = build_logistics_dataframe(engine, master_df=master_df)
    validate_dataframe(logistics_df)

    # Logistics Analysis
    print("=" * 60)
    print("SLA Analysis")
    print("=" * 60)
    contingency, _, _, _,_ = analyze_sla_by_state(logistics_df)

    # RFM Segmentation
    print("=" * 60)
    print("RFM Segmentation")
    print("=" * 60)
    logistics_df, rfm_df, X_train_scaled, X_test_scaled, kmeans = build_customer_segmentation(logistics_df, n_clusters=4)

    # NLP Analysis
    print("=" * 60)
    print("Sentiment Analysis")
    print("=" * 60)
    nlp_df, _, X_sparse, _ = build_sentiment_analysis(logistics_df)

    # Sentiment Model
    print("=" * 60)
    print("Sentiment Model")
    print("=" * 60)
    pipeline, X_test, _, y_test = train_sentiment_model(nlp_df, X_sparse)
    logger.info(f"Model accuracy: {pipeline.score(X_test, y_test):.2%}")

    # HF Comparison
    _, match_rate = compare_with_huggingface_baseline(nlp_df, sample_size=500)
    logger.info(f"HF match: {match_rate:.1%}")

    print("=" * 60)
    print("RAG Assistant")
    print("=" * 60)

    # Businesss documents 
    documents = build_knowledge_base(logistics_df,contingency,nlp_df)

    # Embeddings
    document_vectors = create_embeddings(documents)

    # Vector Data Base 
    collection = create_vector_store(documents,document_vectors)

    # Ask Assistant
    question = "What are the SLA breach rates by state?"
    answer = ask_assistant(question,collection)
    logger.info("Assistant response: %s", answer)

    print("=" * 60)
    print("Pipeline completed successfully")
    print("=" * 60)


if __name__ == "__main__":
    main()