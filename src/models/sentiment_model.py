import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import shap
import logging

logger = logging.getLogger(__name__)
RANDOM_STATE = 42
NUMERIC_FEATURE_COLS = ["is_late", "review_length", "delivery_days"]
SCALED_FEATURE_COLS = ["review_length", "delivery_days"]

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
        Trained model pipeline, test feature matrix, predicted labels,
        actual test labels and SHAP values for model explainability.
    """

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
    # Fit and predict the model
    pipeline_logit.fit(X_train,y_train)
    y_pred = pipeline_logit.predict(X_test)

    # Evaluation and explain
    accuracy = accuracy_score(y_test, y_pred)

    # SHAP Explainer
    X_test_transformed = pipeline_logit.named_steps["preprocessor"].transform(X_test)
    explainer = shap.LinearExplainer(pipeline_logit.named_steps["logistic"], X_test_transformed)
    shap_values = explainer(X_test_transformed)
    logger.info(f"✅ Model trained, Test Accuracy: {accuracy:.4f}")
    logger.info(f"\n{classification_report(y_test, y_pred)}")
    logger.info(f"✅ SHAP values shape: {shap_values.shape}")

    return pipeline_logit, X_test, y_pred, y_test, shap_values



    