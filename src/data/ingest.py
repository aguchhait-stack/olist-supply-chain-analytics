import os
import pandas as pd
from sqlalchemy import create_engine
import kagglehub
import logging
logger = logging.getLogger(__name__)

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
db_path = os.path.join(DATA_DIR, 'ecommerce.db')
engine = create_engine(f'sqlite:///{db_path}')
logger.info(f"Database path: {db_path}")

def ingestion(path: str = DATA_DIR):
    
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
        logger.info(f"Files loading...: {count}")  
        logger.info("Pipeline completed")
        return engine
    except Exception as e:
        print(f"Error: {e}") 
        raise 
        
if __name__ == "__main__":
    ingestion()