import os
import pandas as pd
from sqlalchemy import create_engine
import kagglehub

path = "data"
os.makedirs(path,exist_ok=True)
db_path = os.path.join(path,'ecommerce.db')
engine = create_engine(f'sqlite:///{db_path}')

def ingestion(path=path):
    try:
        path = kagglehub.dataset_download("olistbr/brazilian-ecommerce",output_dir=path,force_download=True)
        count = 0 
        for file in os.listdir(path):
            if file.endswith('.csv'):
                file_path = os.path.join(path,file)
                file = file.replace('olist_','').replace('.csv','').replace('_dataset','')
                file_df = pd.read_csv(file_path)
                count += 1
                file_df.to_sql(f'{file}',con=engine,if_exists='replace',index=False)
        print(f"Files loading: {count}")
        tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", engine)  
        print(f"Tables loaded: {len(tables)}")    
        print("✅ Pipeline completed successfully")
    except Exception as e:
        print(f"❌ Error: {e}") 
        
if __name__ == '__main__':
    ingestion()
