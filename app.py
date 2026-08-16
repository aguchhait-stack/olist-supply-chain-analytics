import streamlit as st
import logging
from main import run_pipeline 
from src.analysis.nlp_analysis import tfidf_vectorizer
from src.visualization.business import plot_monthly_performance_dashboard
from src.visualization.nlp import (plot_nlp_dashboard, 
                                   plot_predictive_word, 
                                   plot_sentiment_correlation, 
                                   plot_shap_summary,
                                   plot_confusion_matrix)
from src.visualization.logistics import plot_sla_breach_by_state, plot_logistics_bottlenecks
from src.visualization.rfm import plot_rfm_segments
from src.rag.assistant import ask_assistant
logger = logging.getLogger(__name__)


# Layout 
st.set_page_config(page_title="Olist Supply Chain Analytics", page_icon= "📊", layout= 'wide')
st.title("📊 Olist Supply Chain Analytics")

# Load all data (cached)
@st.cache_resource
def load_pipeline_data():
    return run_pipeline(include_hf_comparison=False) 
(
    logistics_df,
    contingency,
    chi2,
    p,
    dof,
    rfm_df,
    nlp_df,
    pipeline,
    kmeans,
    collection,
    X_sparse,
    vectorizer, 
    y_pred, 
    y_test,
    shap_values
)  = load_pipeline_data()

# Pages
tab1, tab2, tab3, tab4 = st.tabs(["🏠 Overview", "💬 Sentiment", "🚚 Logistics", "👥 Customer Segments"])
with tab1:
    # Local Filter 
    st.header("Business Overview")

    select_year = st.radio("Select Year", ["All", "Current Year", "Last Year"])
    current_year = logistics_df['order_purchase_timestamp'].dt.year.max()
    last_year = current_year - 1
    if select_year == "Current Year":
        period_logistics_df = logistics_df[logistics_df['order_purchase_timestamp'].dt.year == current_year].copy()
        period_nlp_df = nlp_df[nlp_df['order_purchase_timestamp'].dt.year == current_year].copy()
    elif select_year == "Last Year":
        period_logistics_df = logistics_df[logistics_df['order_purchase_timestamp'].dt.year == last_year].copy()
        period_nlp_df = nlp_df[nlp_df['order_purchase_timestamp'].dt.year == last_year].copy()
    else:
        period_logistics_df = logistics_df.copy()
        period_nlp_df = nlp_df.copy()
    
    # KPI 
    total_orders = period_logistics_df["order_id"].nunique()
    total_revenue = period_logistics_df["total_price"].sum()
    avg_review_score = period_logistics_df["mean_review_score"].mean()
    late_delivery_rate = period_logistics_df["is_late"].mean()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Orders",f"{total_orders:,}")
    col2.metric("Total Revenue", f"R$ {total_revenue/1000000:,.1f}M")
    col3.metric("Avg Review Score", f"{avg_review_score:.2f}")
    col4.metric("Late Delivery Rate", f"{late_delivery_rate:.2%}")

    # Monthly KPIs
    st.subheader("📈 Monthly Performance")
    fig = plot_monthly_performance_dashboard(period_logistics_df)
    st.pyplot(fig)

    st.divider()

    st.subheader("💬 NLP Overview")
    fig1 = plot_nlp_dashboard(period_nlp_df, tfidf_vectorizer)
    st.pyplot(fig1)


with tab2:
    st.header("Sentiment Analysis")

    st.subheader("📊 Model Performance")
    fig2 = plot_confusion_matrix(y_test,y_pred)
    st.pyplot(fig2)

    st.divider()

    st.subheader("📊 Model Explainability (SHAP)")
    fig3 = plot_shap_summary(shap_values, vectorizer)
    st.pyplot(fig3)

    st.divider()

    st.subheader("📊 Predictive Words by Sentiment")
    fig4 = plot_predictive_word(pipeline, vectorizer)
    st.pyplot(fig4)
   

with tab3:
    st.header("SLA Analysis")

    st.subheader("📊 SLA Breach by State")
    fig5 = plot_sla_breach_by_state(logistics_df, contingency, chi2, p, dof)
    st.pyplot(fig5)

    st.divider()

    st.subheader("📊 Logistics Bottlenecks")
    fig6 = plot_logistics_bottlenecks(logistics_df)
    st.pyplot(fig6)

    st.divider()

    st.subheader("📊 Sentiment vs Logistics Correlation")
    fig7 = plot_sentiment_correlation(nlp_df)
    st.pyplot(fig7)

with tab4:
    st.header('RFM Segmentation')
    fig8 = plot_rfm_segments(rfm_df)
    st.pyplot(fig8)

with st.sidebar:
    st.subheader("🤖 Ask the Olist Assistant")
    st.caption("Powered by Gemini 3.5 Flash")

    # Sample Questions
    st.caption("Try asking:")
    sample_questions = ["Which state has the highest SLA breach rate?",
    "Which product categories generate the most revenue?",
    "What are the most important features influencing sentiment predictions?",
    "What does the sentiment analysis tell us about late deliveries?",
    "Which customer segment has the highest value, and how should it be targeted?"]

    selected_prompt = None
    for sample_question in sample_questions:
        if st.button(sample_question, use_container_width=True):
            selected_prompt = sample_question
    st.divider()

    # Chat input
    typed_prompt = st.chat_input("Ask me anything...")

    if typed_prompt:
        selected_prompt = typed_prompt

    # Process question
    if selected_prompt:

        with st.chat_message("user"):
            st.markdown(selected_prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    answer = ask_assistant(
                        selected_prompt,
                        collection
                    )
                except Exception:
                    logger.exception("RAG assistant failed")
                    answer = "Sorry, I couldn't process that question."

            st.markdown(answer)



