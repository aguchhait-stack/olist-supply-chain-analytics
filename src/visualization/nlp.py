import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import seaborn as sns
from src.models.sentiment_model import (NUMERIC_FEATURE_COLS, SCALED_FEATURE_COLS)
import logging
logger = logging.getLogger(__name__) 


def plot_nlp_dashboard(nlp_df: pd.DataFrame, tfidf_vectorizer):
    """
    Plot an NLP dashboard comparing review, delivery, and customer metrics.
    """
    late_freq, _, _ = tfidf_vectorizer(nlp_df[nlp_df["is_late"] == 1]["review_comment_message"])
    ontime_freq, _, _ = tfidf_vectorizer(nlp_df[nlp_df["is_late"] == 0]["review_comment_message"])

    late_top5 = dict(list(late_freq.items())[:5])
    ontime_top5 = dict(list(ontime_freq.items())[:5])

    fig, axis = plt.subplots(4, 2, figsize=(16, 14))

    # Review Score Comparisons
    nlp_df.groupby("is_late")["mean_review_score"].mean().plot(kind='bar', ax=axis[0,0], xlabel="", color='teal')
    axis[0,0].set_xticks([0, 1], ['On-time', 'Late'], rotation=0)
    axis[0,0].set_title('Review Score: On-time vs Late')
    axis[0,0].set_ylabel('Avg Score')
    axis[0,0].grid(True, alpha=0.3)

    nlp_df.groupby("is_interstate")["mean_review_score"].mean().plot(kind='bar', ax=axis[0,1], xlabel="", color='teal')
    axis[0,1].set_xticks([0, 1], ['Intrastate', 'Interstate'], rotation=0)
    axis[0,1].set_title('Review Score: Intrastate vs Interstate')
    axis[0,1].set_ylabel('Avg Score')
    axis[0,1].grid(True, alpha=0.3)

    # Review Length Comparisons
    nlp_df.groupby(nlp_df['mean_review_score'].astype(int))["review_length"].mean().sort_values(ascending=True).\
        plot(kind='barh', ax=axis[1,0], xlabel="", color='teal')
    axis[1,0].set_title('Review Score vs Avg Length')
    axis[1,0].set_ylabel('Avg Score')
    axis[1,0].set_xlabel('Avg Length')
    axis[1,0].grid(True, alpha=0.3)

    nlp_df["review_length"].plot(kind="hist", color='teal', bins=10, ax=axis[1,1])
    axis[1,1].axvline(nlp_df["review_length"].mean(), color='red', linestyle='--', alpha=0.3, label='Mean')
    axis[1,1].set_title('Review Length Distribution')
    axis[1,1].grid(True, alpha=0.3)
    axis[1,1].legend()

    # Late Reviews
    axis[2,0].bar(late_top5.keys(), late_top5.values(), color='red')
    axis[2,0].set_title('Top 5 Words: Late Reviews')
    axis[2,0].set_xlabel('Words')
    axis[2,0].set_ylabel('TF-IDF Frequency')
    axis[2,0].tick_params(axis='x', rotation=45)
    axis[2,0].grid(True, alpha=0.3)

    # On-time Reviews
    axis[2,1].bar(ontime_top5.keys(), ontime_top5.values(), color='green')
    axis[2,1].set_title('Top 5 Words: On-Time Reviews')
    axis[2,1].set_xlabel('Words')
    axis[2,1].set_ylabel('TF-IDF Frequency')
    axis[2,1].tick_params(axis='x', rotation=45)
    axis[2,1].grid(True, alpha=0.3)

    (nlp_df.groupby('customer_segment')["mean_review_score"].mean()).plot(kind='barh', ax=axis[3,0], color='teal')
    axis[3,0].set_title('Review Score by Customer Segment')
    axis[3,0].set_ylabel('')
    axis[3,0].set_xlabel('Avg Score')
    axis[3,0].grid(True, alpha=0.3)
    
    nlp_df['mean_review_score'].astype(int).value_counts(normalize=True).sort_index().plot(
        kind='pie', wedgeprops={'width': 0.6}, autopct='%.1f%%', ax=axis[3,1], 
        startangle=90, textprops={'fontsize': 10, 'weight': 'bold'}
    )
    axis[3,1].set_title('Sentiment Distribution')

    plt.suptitle("NLP Analysis Dashboard", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('outputs/nlp_dashboard.png')
    return fig


def plot_predictive_word(model, vectorizer):
    """
    Visualize the most predictive words for each sentiment class.
    """
    # Feature Coefficients
    coef = model.named_steps['logistic'].coef_
    index = model.named_steps['logistic'].classes_

    feature_names = SCALED_FEATURE_COLS + \
                    [col for col in NUMERIC_FEATURE_COLS if col not in SCALED_FEATURE_COLS] + \
                    vectorizer.get_feature_names_out().tolist()

    coef_df = pd.DataFrame(coef, index=index, columns=feature_names).T

    # Bar Charts
    fig, axis = plt.subplots(1, 3, figsize=(21, 8))
    colormap = {'negative': 'red', 'neutral': 'blue', 'positive': 'green'}
    
    for i, (sentiment, color) in enumerate(colormap.items()):
        top_words = coef_df[sentiment][coef_df[sentiment] > 0].nlargest(10)
        axis[i].barh(top_words.index, top_words.values, color=color)
        axis[i].grid(True, alpha=0.3, axis='x')
        axis[i].set_title(f'{sentiment.capitalize()} Reviews',fontsize=11,fontweight='bold')

    plt.suptitle('Top 10 Predictive Words by Sentiment Class', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('outputs/predictive_words_barchart.png')
    return fig


def plot_sentiment_correlation(nlp_df: pd.DataFrame):
    """
    Plot Spearman correlations between logistics drivers and sentiment metrics.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    sns.heatmap(
        nlp_df[['carrier_transit_days', 'is_interstate', 'distance_km', 'is_late',
                'mean_review_score', 'review_length', 'total_freight_value',
                'total_price', 'vendor_handling_days', 'delivery_days']].corr(method='spearman'),
        cmap='coolwarm', square=True, annot=True, vmin=-1, vmax=1,
        linewidths=0.5, fmt=".2f", ax=ax
    )
    
    ax.set_xticklabels(ax.get_xticklabels(), rotation=50)
    ax.set_title('Spearman Correlation: Logistics Drivers vs Sentiment', 
                 fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('outputs/spearman_correlation_drivers_vs_sentiment.png')
    return fig

def plot_shap_summary(shap_values, vectorizer):
    """
    Plot SHAP feature importance for sentiment prediction.
    """
    feature_names = SCALED_FEATURE_COLS + \
                        [col for col in NUMERIC_FEATURE_COLS if col not in SCALED_FEATURE_COLS] + \
                        vectorizer.get_feature_names_out().tolist()

    shap_importance = abs(shap_values.values).mean(axis=(0, 2))
    top_indices = shap_importance.argsort()[-15:]
    top_features = [feature_names[i] for i in top_indices]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top_features,shap_importance[top_indices], color='teal')
    ax.grid(True, alpha=0.3, axis='x')
    
    ax.set_title("Global SHAP Feature Importance", fontsize=11, fontweight='bold')
    ax.set_xlabel("Mean Absolute SHAP Value")
    plt.tight_layout()
    plt.savefig('outputs/shap_feature_importance_top15.png')
    return fig


def plot_confusion_matrix(y_test, y_pred):
    """
    Plot the confusion matrix for sentiment classification.
    """
    fig, ax = plt.subplots(figsize=(7, 6))

    ConfusionMatrixDisplay.from_predictions(y_test,y_pred,ax=ax,cmap="Blues")
    ax.set_title("Sentiment Classification Confusion Matrix",fontsize=13,fontweight="bold")
    plt.tight_layout()
    plt.savefig("outputs/sentiment_confusion_matrix.png")
    return fig