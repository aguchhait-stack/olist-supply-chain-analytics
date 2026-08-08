from src.data.ingest import engine, ingestion
from src.data.builder import build_logistics_dataframe
from src.analytics.rfm import build_customer_segmentation
from src.nlp.sentiment import build_sentiment_analysis

import pandas as pd


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

    # Build logistics dataframe
    logistics_df = build_logistics_dataframe(engine)

    # Customer segmentation
    logistics_df, rfm, X_train_scaled, X_test_scaled, kmeans =  build_customer_segmentation(logistics_df)


    # NLP
    nlp_df, word_frequency, X_matrix, vectorizer = build_sentiment_analysis(logistics_df)
    


    print("Pipeline completed successfully.")
    return logistics_df, rfm, nlp_df


if __name__ == "__main__":
    main()