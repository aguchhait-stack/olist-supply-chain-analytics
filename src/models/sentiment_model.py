import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression

RANDOM_STATE = 42
NUMERIC_FEATURE_COLS = ["is_late", "review_length", "delivery_days"]
SCALED_FEATURE_COLS = ["review_length", "delivery_days"]

def label_sentiment(nlp_df: pd.DataFrame) -> pd.DataFrame:
    """
    Map review scores to positive, neutral, or negative sentiment labels.
    """
    nlp_df = nlp_df.copy()
    nlp_df['sentiment'] = nlp_df['mean_review_score'].astype(int).map(lambda x: 'positive' if x>=4  
                                                                            else 'neutral' if  x ==3  
                                                                            else 'negative')
    return nlp_df

def _build_feature_matrix(nlp_df: pd.DataFrame, X_sparse: csr_matrix) -> tuple:
    """
    Combine numeric features with the TF-IDF sparse matrix.
    """

    # numeric columns converted to sparse via csr_matrix, combining with TF-IDF sparse matrix with hstack
    features = hstack([csr_matrix(nlp_df[NUMERIC_FEATURE_COLS].to_numpy()), X_sparse])

    # target
    target = nlp_df['sentiment'].to_numpy()
    return features, target


def train_sentiment_model(nlp_df: pd.DataFrame, X_sparse: csr_matrix, max_iter = 120,regularization = 0.1) -> tuple:

    """
    Train a Logistic Regression model for review sentiment classification.

    Parameters
    ----------
    nlp_df : pd.DataFrame
        NLP dataframe containing review scores and numeric features.

    X_sparse : csr_matrix
        Sparse TF-IDF feature matrix generated during NLP preprocessing.

    max_iter : int, default=120
        Maximum number of iterations for Logistic Regression convergence.

    regularization : float, default=0.1
        Inverse regularization strength passed to Logistic Regression.

    Returns
    -------
    tuple
        Trained Logistic Regression pipeline, predicted labels,
        and test-set labels.
    """

    nlp_df = label_sentiment(nlp_df)
    features, target = _build_feature_matrix(nlp_df,X_sparse)

    # train test split, stratify for biasness proof
    X_train, X_test, y_train, y_test = train_test_split(features,target,test_size=0.2,random_state= RANDOM_STATE,shuffle=True, stratify=target)

    # Extracting columns index for scaled features
    col_index = [index for index,col in enumerate(NUMERIC_FEATURE_COLS) if col in SCALED_FEATURE_COLS]

    # Scaling with 'review_length', 'delivery_days' due to positive skewness
    preprocessor = ColumnTransformer([('scaler',RobustScaler(with_centering=False),col_index)],remainder='passthrough')

   

    # Choosen 'saga' solver for fast convergence for multi-class classification
    # 67.4% positive vs 8.9% neutral class imbalance penelizing by class weight
    pipeline_logit = Pipeline([('preprocessor',preprocessor),('logistic',LogisticRegression(class_weight='balanced',
                                                        max_iter = max_iter, # iteration for opimal solution
                                                        random_state= RANDOM_STATE,
                                                        solver = 'saga',
                                                        C = regularization # stronger regularization for rare words
                                                        ))])
    pipeline_logit.fit(X_train,y_train)
    y_pred = pipeline_logit.predict(X_test)
    return pipeline_logit, X_test, y_pred, y_test



    