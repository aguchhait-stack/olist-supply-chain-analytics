import pandas as pd
from transformers import pipeline
from sklearn.metrics import classification_report
import warnings
import logging
import os

logger = logging.getLogger(__name__)

#Supress warning
warnings.filterwarnings('ignore')
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
HF_MODEL_NAME = 'cardiffnlp/xlm-roberta-base-tweet-sentiment-pt'
SAMPLE_SIZE = 500
RANDOM_STATE = 42

def compare_with_huggingface_baseline(nlp_df: pd.DataFrame, sample_size: int = SAMPLE_SIZE) -> tuple:

    pipe = pipeline(task="sentiment-analysis",model = HF_MODEL_NAME,truncation=True)

    # sample size for testing 
    sample = nlp_df.sample(sample_size, random_state= RANDOM_STATE)
    sample['sentiment_hf'] = [result['label'] for result in  pipe(sample["review_comment_message"].tolist())]

    # Comparing with TD-IDF Based sentiment analysis
    report = classification_report(sample['sentiment'], sample['sentiment_hf'])
    match_rate = (sample['sentiment'] == sample['sentiment_hf']).mean()
    logger.info("HF baseline match rate vs TF-IDF labels: %.1f%%", match_rate * 100)
    
    return report, match_rate