import streamlit as st
import os
import logging 
from src.data.ingest import engine, ingestion
from src.data.logistics_data import build_logistics_dataframe
from src.visualization.business import plot_monthly_performance_dashboard
logger = logging.getLogger(__name__)

# Layout
st.set_page_config(page_title="Olist Supply Chain Analytics", page_icon= "📊", layout= 'centered')
st.title("📊 Olist Supply Chain Analytics")

PROCESSED_DATA_DIR = "data/processed/ecommerce.db"

@st.cache_data
def check_and_build_database(dir = PROCESSED_DATA_DIR):

    if not os.path.exists(dir):
        st.warning("Building database(3-5 minutes)...")
        ingestion()
    return build_logistics_dataframe(engine)

logistics_df = check_and_build_database()


page = st.sidebar.radio("Go to:",["Overview", "Logistics", "RFM", "NLP", "RAG"])
if page == "Overview":
    st.header("Business Overview")

    # KPI 
    total_orders = logistics_df["order_id"].nunique()
    total_revenue = logistics_df["total_price"].sum()
    avg_review_score = logistics_df["mean_review_score"].mean()
    late_delivery_rate = logistics_df["is_late"].mean()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Orders",f"{total_orders:,}")
    col2.metric("Total Revenue", f"R$ {total_revenue/1000000:,.1f}M")
    col3.metric("Avg Review Score", f"{avg_review_score:.2f}")
    col4.metric("Late Delivery Rate", f"{late_delivery_rate:.2%}")

    # Monthly KPIs
    st.subheader("📈 Monthly Performance")
    fig = plot_monthly_performance_dashboard(logistics_df)
    st.pyplot(fig)