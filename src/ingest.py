import os
import pandas as pd
from sqlalchemy import create_engine
import kagglehub

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
db_path = os.path.join(DATA_DIR, 'ecommerce.db')
engine = create_engine(f'sqlite:///{db_path}')

def ingestion(path: str = DATA_DIR):
    """
    Download the Olist dataset from Kaggle and load CSV files into SQLite.

    This function downloads the official Olist Brazilian e-commerce dataset
    using `kagglehub`, creates a SQLite database, and loads each CSV file
    into a separate SQL table. Table names are derived from the filenames by
    removing the ``olist_`` prefix and ``_dataset`` suffix.

    Args:
        path (str, optional): Directory where the dataset and SQLite database
            will be stored. Defaults to ``"data"``.

    Returns:
        Engine: SQLAlchemy engine connected to the SQLite database.

    Raises:
    Exception: If downloading the dataset or loading the CSV files into
        the SQLite database fails.
    """
    try:
        dataset_path = kagglehub.dataset_download("olistbr/brazilian-ecommerce",output_dir=path,force_download=True)
        count = 0 
        for file in os.listdir(dataset_path):
            if file.endswith('.csv'):
                file_path = os.path.join(dataset_path,file)
                file = file.replace('olist_','').replace('.csv','').replace('_dataset','')
                file_df = pd.read_csv(file_path)
                count += 1
                file_df.to_sql(f'{file}',con=engine,if_exists='replace',index=False)
        print(f"Files loading: {count}")
        tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", engine)  
        print(f"Tables loaded: {len(tables)}")    
        print("✅ Pipeline completed successfully")
        return engine
    except Exception as e:
        print(f"❌ Error: {e}") 
        raise 
        
if __name__ == '__main__':
    ingestion()
