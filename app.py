import streamlit as st
import logging
from main import run_pipeline 
from src.analysis.nlp_analysis import tfidf_vectorizer
from src.visualization.business import plot_monthly_performance_dashboard
from src.visualization.nlp import plot_nlp_dashboard, plot_predictive_word, plot_sentiment_correlation
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
logistics_df,contingency,chi2,p,dof,rfm_df,nlp_df,pipeline,kmeans,collection,X_sparse,vectorizer = load_pipeline_data()

# Pages
tab1, tab2, tab3, tab4 = st.tabs(["🏠 Overview", "💬 Review KPI", "🚚 Logistics KPI", "👥 Purchase Behavior"])
with tab1:
    # Local Filter 
    st.header("Business Overview")
    select_year = st.radio("Select Year", ["All", "Current Year", "Last Year"])
    current_year = logistics_df['order_purchase_timestamp'].dt.year.max()
    last_year = current_year - 1
    if select_year == "Current Year":
        period_df = logistics_df[logistics_df['order_purchase_timestamp'].dt.year == current_year].copy()
    elif select_year == "Last Year":
        period_df = logistics_df[logistics_df['order_purchase_timestamp'].dt.year == last_year].copy()
    else:
        period_df = logistics_df.copy()
    
    # KPI 
    total_orders = period_df["order_id"].nunique()
    total_revenue = period_df["total_price"].sum()
    avg_review_score = period_df["mean_review_score"].mean()
    late_delivery_rate = period_df["is_late"].mean()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Orders",f"{total_orders:,}")
    col2.metric("Total Revenue", f"R$ {total_revenue/1000000:,.1f}M")
    col3.metric("Avg Review Score", f"{avg_review_score:.2f}")
    col4.metric("Late Delivery Rate", f"{late_delivery_rate:.2%}")

    # Monthly KPIs
    st.subheader("📈 Monthly Performance")
    fig = plot_monthly_performance_dashboard(period_df)
    st.pyplot(fig)

with st.sidebar:
    st.subheader("Ask the Olist Assistant")
    
    # Sample Questions as Buttons
    st.caption("Try asking:")
    
    if st.button("What is the late delivery rate?", use_container_width=True):
        prompt = "What is the late delivery rate?"
    
    if st.button("Show top 5 products by revenue", use_container_width=True):
        prompt = "Show top 5 products by revenue"
    
    if st.button("Which state has the worst SLA?", use_container_width=True):
        prompt = "Which state has the worst SLA breach rate?"
    
    if st.button("How is customer sentiment overall?", use_container_width=True):
        prompt = "How is customer sentiment overall?"
    
    if st.button("What are the customer segments?", use_container_width=True):
        prompt = "What are the customer segments and their characteristics?"
    
    if st.button("What is the average delivery time?", use_container_width=True):
        prompt = "What is the average delivery time in days?"
    
    st.divider()
    
    # Chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for msg in st.session_state.messages[-10:]:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask me anything..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    answer = ask_assistant(prompt, collection)
                except Exception as e:
                    answer = f"Error: {str(e)}"
            st.write(answer)
        
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.rerun()
with tab2:
    st.header("Sentiment Analysis")
    fig1 = plot_nlp_dashboard(nlp_df, tfidf_vectorizer)
    st.pyplot(fig1)
    fig2 = plot_predictive_word(pipeline, vectorizer)
    st.pyplot(fig2)

with tab3:
    st.header("SLA Analysis")
    fig1 = plot_sla_breach_by_state(logistics_df, contingency, chi2, p, dof)
    st.pyplot(fig1)
    fig2 = plot_logistics_bottlenecks(logistics_df)
    st.pyplot(fig2)
    fig3 = plot_sentiment_correlation(nlp_df)
    st.pyplot(fig3)

with tab4:
    st.header('RFM Segmentation')
    fig1 = plot_rfm_segments(rfm_df)
    st.pyplot(fig1)



