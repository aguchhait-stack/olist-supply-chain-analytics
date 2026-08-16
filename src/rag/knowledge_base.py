import pandas as pd
import logging
from src.models.sentiment_model import (NUMERIC_FEATURE_COLS, SCALED_FEATURE_COLS)


logger = logging.getLogger(__name__)


def build_knowledge_base(logistics_df: pd.DataFrame,
                         contingency: pd.DataFrame,
                         nlp_df: pd.DataFrame,
                         rfm_df: pd.DataFrame,
                         shap_values,
                         vectorizer) -> list[str]:

    insights = []

    # Business KPI
    total_orders = logistics_df["order_id"].nunique()
    total_revenue = logistics_df["total_price"].sum()
    avg_order_value = total_revenue / total_orders if total_orders > 0 else 0
    avg_review_score = logistics_df["mean_review_score"].mean()
    late_rate = logistics_df["is_late"].mean()
    min_date = logistics_df['order_purchase_timestamp'].min()
    max_date = logistics_df['order_purchase_timestamp'].max()

    insights.append(f"Total Revenue: R$ {total_revenue:,.2f}")
    insights.append(f"Total Orders: {total_orders:,}")
    insights.append(f"Average Order Value: R$ {avg_order_value:,.2f}")
    insights.append(f"Average Review Score: {avg_review_score:.2f} / 5.0")
    insights.append(f"Late Delivery Rate: {late_rate:.2%}")
    insights.append(f"Data Period: {min_date.date()} to {max_date.date()}")
    top_categories = (logistics_df.groupby("product_category")["total_price"].sum().nlargest(5))
    insights.append("Top 5 Product Categories by Revenue:")
    for category, value in top_categories.items():
        insights.append(f"  {category}: R$ {value:,.0f}")
    insights.append("""
    Revenue and orders peak in Nov 2017 during Black Friday — clear seasonality.
    Delivery days improved from 50+ days in 2016 to below 10 days by 2017 — rapid operational maturity.
    Late rate spikes during peak seasons like Nov 2017 — suggesting carrier capacity constraints during high demand.
    Top 5 categories drive ~40% of total revenue, with computers & accessories leading.
    Review score stabilised around 4.1 - 4.3 as operations matured.
    """)

    # Logistics
    interstate_rate = logistics_df[logistics_df["is_interstate"] == 1]["is_late"].mean()
    intrastate_rate = logistics_df[logistics_df["is_interstate"] == 0]["is_late"].mean()
    correlation_cols = ["is_late", "carrier_transit_days", "vendor_handling_days", "distance_km", "delivery_days"]
    correlation = logistics_df[correlation_cols].corr(method="spearman")["is_late"].drop("is_late")
    strongest_driver = correlation.abs().idxmax()
    strongest_value = correlation[strongest_driver]
    insights.append("Logistics Performance:")
    insights.append(f"Strongest driver of late delivery: {strongest_driver} (Spearman correlation = {strongest_value:.2f}).")
    insights.append(f"  Average Delivery Time: {logistics_df['delivery_days'].mean():.1f} days")
    insights.append(f"  Average Distance: {logistics_df['distance_km'].mean():.1f} km")
    insights.append(f"  Carrier transit: {logistics_df['carrier_transit_days'].mean():.1f} days")
    insights.append(f"  Vendor handling: {logistics_df['vendor_handling_days'].mean():.1f} days")
    sla_summary = (contingency[1] * 100 / contingency.sum(axis=1)).sort_values(ascending=False)
    sla_summary_text = ", ".join(f"{state}: {rate:.2f}%" for state, rate in sla_summary.items())
    insights.append(f"SLA breach rates by state: {sla_summary_text}")
    insights.append(f"Late delivery rate: {logistics_df['is_late'].mean():.2%}")
    insights.append(f"Interstate orders have a {interstate_rate:.1%} late delivery rate, compared to {intrastate_rate:.1%} for intrastate orders.")
    insights.append("Chi-square analysis (χ² = 1622.01, p < 0.001) confirms that SLA breach rates vary significantly across Brazilian states")
    

    # RFM
    segment_profiles = rfm_df.groupby("KMeans_Segment_name").agg(
        customer_count=("customer_unique_id", "count"),
        avg_recency=("recency", "mean"),
        avg_frequency=("frequency", "mean"),
        avg_monetary=("monetary", "mean"))

    insights.append("Purchase Behaviour of Customers:")
    for segment, row in segment_profiles.iterrows():
        insights.append(f"{segment}: {row['customer_count']:,} customers, average recency {row['avg_recency']:.0f} days,"
            f"average frequency {row['avg_frequency']:.1f} orders, average monetary value R${row['avg_monetary']:.2f}.")
    insights.append("""
    Champions segment customers drive the highest value, need loyalty discounts.
    "At risk" customers represent high-spend one-time buyers, need re-engagement before they become Lost.
    """)

    # NLP
    insights.append(f"Average review score is {nlp_df['mean_review_score'].mean():.2f}")
    insights.append(f"Total reviews: {len(nlp_df):,}")
    late_review_gap = (nlp_df.groupby("is_late")["mean_review_score"].mean()[0]- nlp_df.groupby("is_late")["mean_review_score"].mean()[1])
    insights.append(
        f"Late orders have lower review scores than on-time orders "
        f"(gap of {late_review_gap:.2f} points).")
    insights.append(f"""
    Customers write longer reviews when they have a bad experience (e.g. ~100 characters for a 1-star review).
    Review score shows limited variation across customer segments and interstate status,
    so these were excluded from the sentiment model's feature set.
    """)
    for sentiment, pct in nlp_df["sentiment"].value_counts(normalize=True).mul(100).items():
        insights.append(f"{pct:.1f}% of customer reviews are classified as {sentiment}.")
    insights.append(f"""
    Carrier transit ({logistics_df['carrier_transit_days'].mean():.1f} days) is longer than vendor handling
    ({logistics_df['vendor_handling_days'].mean():.1f} days) — bottleneck is the carrier, not the seller.
    Longer-distance shipments generally tend to have higher freight costs.
    {strongest_driver} (Spearman = {strongest_value:.2f}) is the strongest driver of late deliveries,
    indicating regional carrier optimization should be the top logistics priority.
    """)
    # Model Explainability
    feature_names = SCALED_FEATURE_COLS + \
                        [col for col in NUMERIC_FEATURE_COLS if col not in SCALED_FEATURE_COLS] + \
                        vectorizer.get_feature_names_out().tolist()
    shap_importance = abs(shap_values.values).mean(axis=(0, 2))
    top_indices = shap_importance.argsort()[-15:][::-1]

    insights.append("Model Explainability - Global SHAP Feature Importance:")
    for index in top_indices:
        insights.append(f"  {feature_names[index]}: {shap_importance[index]:.4f}")
    insights.append(
            "SHAP values show the overall importance of features "
            "across sentiment model predictions. The model uses both "
            "review text and operational features such as review length, "
            "late delivery, and delivery time.")

    logger.info("Knowledge base created: %d documents", len(insights))

    return insights
