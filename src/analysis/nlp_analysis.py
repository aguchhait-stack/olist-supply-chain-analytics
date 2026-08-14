import pandas as pd
from string import punctuation
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import RSLPStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import logging
import nltk 
# Download once
#nltk.download('punkt') 
#nltk.download('stopwords')
#nltk.download('punkt_tab')
#nltk.download('rslp')
logger = logging.getLogger(__name__)

def _build_nlp_dataframe(logistics_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create the NLP dataframe from customer review comments.
    """
    nlp_df = logistics_df[logistics_df["review_comment_message"].notna()].copy()
    nlp_df['review_length'] = nlp_df["review_comment_message"].str.len()
    logger.info(f"✅ NLP dataframe: {nlp_df.shape[0]:,} reviews")
    return nlp_df

def _label_sentiment(nlp_df: pd.DataFrame) -> pd.DataFrame:
    """
    Map review scores to positive, neutral, or negative sentiment labels.
    """
    nlp_df = nlp_df.copy()
    nlp_df['sentiment'] = nlp_df['mean_review_score'].astype(int).map(lambda x: 'positive' if x>=4  
                                                                            else 'neutral' if  x ==3  
                                                                            else 'negative')
    logger.info(f"Avg score: {nlp_df['mean_review_score'].mean():.2f}")
    return nlp_df


def tfidf_vectorizer(column: pd.Series) -> tuple:
    """
    Process Portuguese review text using TF-IDF.

    Returns
    -------
    tuple
        Word frequency dictionary, sparse TF-IDF matrix,
        and fitted TF-IDF vectorizer.
    """
    # set of portugease stopwords and punctuation
    punc = punctuation
    pt_stopwords = stopwords.words('portuguese')
    custom_fillers = {"pra", "q", "pro", "produto", "comprei", "o", "a", "da"} 
    normalizer = set(pt_stopwords).union(punc).union(custom_fillers)

    # Word Tokenization 
    word_token =  [word_tokenize(word) for word in column.astype(str)]
    word_token_normalized = [[word.lower() for word in doc if word.lower() not in normalizer]for doc in word_token]

    # stemmer object for Portuguese
    stemmer = RSLPStemmer()
    stemmed_word_token_normalized = [' '.join([stemmer.stem(word) for word in doc])for doc in word_token_normalized]

    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer()
    X_sparse = vectorizer.fit_transform(stemmed_word_token_normalized)
    feature_name = vectorizer.get_feature_names_out()

    # TF-IDF Words Frequency
    word_frequency = dict(zip(feature_name,X_sparse.toarray().sum(axis=0)))
    word_frequency_sorted = dict(sorted(word_frequency.items(), key = lambda x: x[1], reverse=True))
    logger.info(f"✅ TF-IDF complete: {len(feature_name):,} features")
    return word_frequency_sorted, X_sparse, vectorizer

def build_sentiment_analysis(logistics_df: pd.DataFrame) -> tuple:
    """
    Build the NLP dataset and TF-IDF representation.

    Pipeline
    --------
    1. Create the NLP dataframe.
    2. Vectorize review comments using TF-IDF.

    Returns
    -------
    tuple
        NLP dataframe, word-frequency dictionary,
        sparse TF-IDF matrix, and fitted vectorizer.
    """
    nlp_df = _build_nlp_dataframe(logistics_df)
    nlp_df = _label_sentiment(nlp_df)
    word_frequency_sorted, X_sparse, vectorizer = tfidf_vectorizer(nlp_df['review_comment_message'])
    return nlp_df,word_frequency_sorted, X_sparse, vectorizer