import streamlit as st
import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from src.rag.assistant import ask_assistant
from src.rag.vector_store import get_collection


# ============================================================
# Page configuration
# ============================================================

st.set_page_config(
    page_title="Olist Business Assistant",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# Load existing data
# ============================================================

@st.cache_data
def load_data():

    # Use your existing processed datasets here.
    # Replace these imports/loads with however you currently
    # create logistics_df, nlp_df and rfm in your project.

    return logistics_df, nlp_df, rfm


# ============================================================
# Load Chroma collection
# ============================================================

@st.cache_resource
def load_collection():
    return get_collection()


collection = load_collection()


# ============================================================
# Title
# ============================================================

st.title("📊 Olist Supply Chain Analytics")

st.caption(
    "Business performance, logistics, customer analytics and AI assistant"
)


# ============================================================
# Tabs
# ============================================================

tab_dashboard, tab_logistics, tab_customer, tab_assistant = st.tabs(
    [
        "📊 Executive Dashboard",
        "🚚 Logistics",
        "👥 Customer",
        "💬 Business Assistant"
    ]
)


# ============================================================
# EXECUTIVE DASHBOARD
# ============================================================

with tab_dashboard:

    st.header("Executive Dashboard")

    # --------------------------------------------------------
    # KPI calculations
    # --------------------------------------------------------

    total_revenue = logistics_df["total_price"].sum()

    total_orders = logistics_df["order_id"].nunique()

    average_review = logistics_df["mean_review_score"].mean()

    late_rate = (
        logistics_df["is_late"].mean() * 100
    )

    average_delivery = (
        logistics_df["delivery_days"].mean()
    )

    vendor_days = (
        logistics_df["vendor_handling_days"].mean()
    )

    carrier_days = (
        logistics_df["carrier_transit_days"].mean()
    )

    carrier_vendor_ratio = (
        carrier_days / vendor_days
    )


    # --------------------------------------------------------
    # KPI cards
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Revenue",
        f"R$ {total_revenue / 1_000_000:.1f}M"
    )

    col2.metric(
        "Orders",
        f"{total_orders:,}"
    )

    col3.metric(
        "Average Review",
        f"{average_review:.2f} / 5"
    )

    col4.metric(
        "Late Delivery Rate",
        f"{late_rate:.1f}%"
    )


    col5, col6 = st.columns(2)

    col5.metric(
        "Average Delivery",
        f"{average_delivery:.1f} days"
    )

    col6.metric(
        "Carrier / Vendor Time",
        f"{carrier_vendor_ratio:.1f}×"
    )


    st.divider()


    # --------------------------------------------------------
    # Monthly performance
    # --------------------------------------------------------

    st.subheader("Monthly Performance")

    monthly = (
        logistics_df
        .set_index("order_purchase_timestamp")
        .resample("ME")
        .agg(
            revenue=("total_price", "sum"),
            orders=("order_id", "nunique"),
            late_rate=("is_late", "mean"),
            review_score=("mean_review_score", "mean")
        )
    )

    monthly["late_rate"] *= 100


    col1, col2 = st.columns(2)


    # Revenue
    with col1:

        st.write("Monthly Revenue")

        st.line_chart(
            monthly["revenue"]
        )


    # Orders
    with col2:

        st.write("Monthly Orders")

        st.line_chart(
            monthly["orders"]
        )


    col1, col2 = st.columns(2)


    # Review score
    with col1:

        st.write("Average Review Score")

        st.line_chart(
            monthly["review_score"]
        )


    # Late rate
    with col2:

        st.write("Late Delivery Rate")

        st.line_chart(
            monthly["late_rate"]
        )


# ============================================================
# LOGISTICS
# ============================================================

with tab_logistics:

    st.header("Logistics Performance")


    # --------------------------------------------------------
    # Carrier vs vendor
    # --------------------------------------------------------

    st.subheader(
        "Vendor Handling vs Carrier Transit"
    )

    logistics_time = pd.Series(
        {
            "Vendor Handling": logistics_df[
                "vendor_handling_days"
            ].mean(),

            "Carrier Transit": logistics_df[
                "carrier_transit_days"
            ].mean()
        }
    )

    st.bar_chart(
        logistics_time
    )


    st.write(
        f"Average vendor handling time: "
        f"{vendor_days:.1f} days"
    )

    st.write(
        f"Average carrier transit time: "
        f"{carrier_days:.1f} days"
    )


    st.divider()


    # --------------------------------------------------------
    # SLA breach by state
    # --------------------------------------------------------

    st.subheader(
        "SLA Breach Rate by State"
    )

    state_sla = (
        logistics_df
        .groupby("customer_state")["is_late"]
        .mean()
        .mul(100)
        .sort_values(ascending=False)
    )

    st.bar_chart(
        state_sla
    )


    # --------------------------------------------------------
    # Top 5 states
    # --------------------------------------------------------

    st.subheader(
        "Top 5 States with Highest SLA Breach Rate"
    )

    top_states = (
        state_sla
        .head(5)
        .round(2)
        .rename("SLA Breach Rate (%)")
    )

    st.dataframe(
        top_states,
        use_container_width=True
    )


# ============================================================
# CUSTOMER
# ============================================================

with tab_customer:

    st.header("Customer Analytics")


    # --------------------------------------------------------
    # Review score: late vs on-time
    # --------------------------------------------------------

    st.subheader(
        "Review Score: Late vs On-Time Orders"
    )

    review_by_delivery = (
        nlp_df
        .groupby("is_late")["mean_review_score"]
        .mean()
    )

    review_by_delivery.index = [
        "On-time",
        "Late"
    ]

    st.bar_chart(
        review_by_delivery
    )


    st.divider()


    # --------------------------------------------------------
    # Customer segments
    # --------------------------------------------------------

    st.subheader(
        "Customer Segments"
    )

    segment_counts = (
        rfm["KMeans_Segment_name"]
        .value_counts()
    )

    st.bar_chart(
        segment_counts
    )


    # --------------------------------------------------------
    # RFM scatter
    # --------------------------------------------------------

    st.subheader(
        "RFM Customer Segments"
    )

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    for segment in rfm[
        "KMeans_Segment_name"
    ].unique():

        subset = rfm[
            rfm["KMeans_Segment_name"] == segment
        ]

        ax.scatter(
            subset["recency"],
            subset["monetary"],
            s=subset["frequency"] * 20,
            alpha=0.4,
            label=segment
        )

    ax.set_xlabel(
        "Recency (days)"
    )

    ax.set_ylabel(
        "Monetary Value (R$)"
    )

    ax.set_title(
        "Customer Segmentation"
    )

    ax.legend()

    ax.grid(alpha=0.3)

    st.pyplot(fig)


# ============================================================
# BUSINESS ASSISTANT
# ============================================================

with tab_assistant:

    st.header(
        "💬 Olist Business Assistant"
    )

    st.caption(
        "Ask questions about business performance, "
        "logistics and customers."
    )


    # --------------------------------------------------------
    # Quick questions
    # --------------------------------------------------------

    sample_questions = [
        "What's the total revenue?",
        "What's the late delivery rate?",
        "What's the average review score?",
        "Which state has the highest SLA breach rate?",
        "What drives late deliveries?"
    ]


    with st.sidebar:

        st.subheader(
            "Quick Questions"
        )

        for question in sample_questions:

            if st.button(
                question,
                use_container_width=True
            ):

                st.session_state.question = question


        st.divider()

        st.caption(
            "RAG · Gemini · ChromaDB"
        )


    # --------------------------------------------------------
    # Chat input
    # --------------------------------------------------------

    question = st.chat_input(
        "Ask a business question..."
    )


    if (
        "question" in st.session_state
        and not question
    ):

        question = st.session_state.question

        st.session_state.question = ""


    # --------------------------------------------------------
    # Answer
    # --------------------------------------------------------

    if question:

        with st.chat_message("user"):

            st.write(question)


        with st.chat_message("assistant"):

            with st.spinner(
                "Searching knowledge base..."
            ):

                try:

                    answer = ask_assistant(
                        question=question,
                        collection=collection
                    )

                    st.write(answer)

                except Exception as e:

                    st.error(
                        f"Error: {e}"
                    )