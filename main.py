from src.ingest import ingestion
from src.queries import get_top_revenue_categories
from src.plots import plot_top_revenue_categories
        
if __name__ == '__main__':
    
    # Ingestion from Kaggle API to local SQLITE database
    engine = ingestion()
    data = get_top_revenue_categories(engine)
    plot_top_revenue_categories(data)


    
