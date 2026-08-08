import pandas as pd
import matplotlib.pyplot as plt

def plot_nlp_dashboard(nlp_df: pd.DataFrame, tfidf_vectorizer):
    late_freq, _, _ = tfidf_vectorizer(nlp_df[nlp_df["is_late"] ==1]["review_comment_message"])
    ontime_freq, _, _ = tfidf_vectorizer(nlp_df[nlp_df["is_late"] ==0]["review_comment_message"])

    # Late Reviews
    late_top5 = dict(list(late_freq.items())[:5])
    # On-time Reviews
    ontime_top5 = dict(list(ontime_freq.items())[:5])

    fig, axis = plt.subplots(4,2,figsize = (16,14))

    # Review Score Comparisons
    nlp_df.groupby("is_late")["mean_review_score"].mean().plot(kind='bar',ax=axis[0,0],xlabel="",color='teal')
    axis[0,0].set_xticks([0, 1], ['On-time', 'Late'], rotation=0)
    axis[0,0].set_title('Review Score: On-time vs Late')
    axis[0,0].set_ylabel('Avg Score')
    axis[0,0].grid(True, alpha=0.3)

    nlp_df.groupby("is_interstate")["mean_review_score"].mean().plot(kind='bar',ax=axis[0,1],xlabel="",color='teal')
    axis[0,1].set_xticks([0, 1], ['Intrastate', 'Interstate'], rotation=0)
    axis[0,1].set_title('Review Score: Intrastate vs Interstate')
    axis[0,1].set_ylabel('Avg Score')
    axis[0,1].grid(True, alpha=0.3)

    # Review Length Comparisons
    nlp_df.groupby(nlp_df['mean_review_score'].astype(int))["review_length"].mean().sort_values(ascending=True).\
        plot(kind='barh',ax=axis[1,0],xlabel="",color='teal')
    axis[1,0].set_title('Review Score vs Avg Length')
    axis[1,0].set_ylabel('Avg Score')
    axis[1,0].set_xlabel('Avg Length')
    axis[1,0].grid(True, alpha=0.3)

    nlp_df["review_length"].plot(kind="hist",color='teal',bins=10,ax=axis[1,1])
    axis[1,1].axvline(nlp_df["review_length"].mean(),color='red',linestyle='--',alpha=0.3,label= 'Mean')
    axis[1,1].set_title('Review Length Distribution')
    axis[1,1].grid(True, alpha=0.3)
    axis[1,1].legend()

    # Late Reviews
    axis[2,0].bar(late_top5.keys(),late_top5.values(),color='red')
    axis[2,0].set_title('Top 5 Words: Late Reviews')
    axis[2,0].set_xlabel('Words')
    axis[2,0].set_ylabel('TF-IDF Frequency')
    axis[2,0].tick_params(axis='x', rotation=45)
    axis[2,0].grid(True, alpha=0.3)

    # On-time Reviews
    axis[2,1].bar(ontime_top5.keys(),ontime_top5.values(),color='green')
    axis[2,1].set_title('Top 5 Words: On-Time Reviews')
    axis[2,1].set_xlabel('Words')
    axis[2,1].set_ylabel('TF-IDF Frequency')
    axis[2,1].tick_params(axis='x', rotation=45)
    axis[2,1].grid(True, alpha=0.3)

    (nlp_df.groupby('customer_segment')["mean_review_score"].mean()).plot(kind='barh',ax=axis[3,0],color='teal')
    axis[3,0].set_title('Review Score by Customer Segment')
    axis[3,0].set_ylabel('')
    axis[3,0].set_xlabel('Avg Score')
    axis[3,0].grid(True, alpha=0.3)
    nlp_df['mean_review_score'].astype(int).value_counts(normalize=True).sort_index().plot(kind='pie',wedgeprops= {'width':0.6},
                                                                            autopct='%.1f%%',ax=axis[3,1],startangle = 90,
                                                                            textprops={'fontsize': 10, 'weight': 'bold'})
    axis[3,1].set_title('Sentiment Distribution')


    plt.suptitle("NLP Analysis Dashboard", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('../outputs/nlp_dashboard.png')
    plt.show()