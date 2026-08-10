import pandas as pd
from scipy.stats import chi2_contingency


def build_knowledge_base(
    logistics_df: pd.DataFrame,
    rfm: pd.DataFrame,
    nlp_df: pd.DataFrame
) -> list[str]:
    """
    Build business knowledge documents from existing analysis.

    The knowledge base contains:
    - Business KPIs
    - Logistics metrics
    - Statistical findings
    - Logistics bottleneck findings
    - Freight and distance findings
    - Customer segmentation / RFM findings
    - NLP and customer sentiment findings
    """

    documents = []

    # ============================================================
    # Business performance
    # ============================================================

    visualization_df = (
        logistics_df
        .set_index("order_purchase_timestamp")
        .copy()
    )

    # Monthly revenue
    monthly_revenue = (
        visualization_df["total_price"]
        .resample("ME")
        .sum()
    )

    for date, revenue in monthly_revenue.items():
        documents.append(
            f"Monthly revenue for {date.strftime('%B %Y')} "
            f"was R$ {revenue:,.2f}."
        )

    # Monthly orders
    monthly_orders = (
        visualization_df["order_id"]
        .resample("ME")
        .nunique()
    )

    for date, orders in monthly_orders.items():
        documents.append(
            f"Monthly orders for {date.strftime('%B %Y')} "
            f"were {orders:,}."
        )

    # Monthly freight
    monthly_freight = (
        visualization_df["total_freight_value"]
        .resample("ME")
        .sum()
    )

    for date, freight in monthly_freight.items():
        documents.append(
            f"Monthly freight value for {date.strftime('%B %Y')} "
            f"was R$ {freight:,.2f}."
        )

    # Top 5 product categories
    top5_categories = (
        visualization_df
        .groupby("product_category")["total_price"]
        .sum()
        .nlargest(5)
    )

    for category, revenue in top5_categories.items():
        documents.append(
            f"Product category {category} generated "
            f"R$ {revenue:,.2f} in total revenue."
        )

    # ============================================================
    # Logistics metrics
    # ============================================================

    average_delivery_days = (
        logistics_df["delivery_days"].mean()
    )

    late_rate = (
        logistics_df["is_late"].mean()
    )

    average_freight = (
        logistics_df["total_freight_value"].mean()
    )

    documents.append(
        f"Average delivery time is "
        f"{average_delivery_days:.2f} days."
    )

    documents.append(
        f"Late delivery rate is "
        f"{late_rate:.2%}."
    )

    documents.append(
        f"Average freight value is "
        f"R$ {average_freight:.2f}."
    )

    # ============================================================
    # SLA statistical analysis
    # ============================================================

    contingency = pd.crosstab(
        logistics_df["customer_state"],
        logistics_df["is_late"]
    )

    chi2, p, dof, expected = chi2_contingency(
        contingency
    )

    if p < 0.001:
        p_text = "p < 0.001"
    else:
        p_text = f"p = {p:.4f}"

    documents.append(
        f"SLA breach rates vary significantly across Brazilian "
        f"states (chi-square = {chi2:.2f}, {p_text}, "
        f"dof = {dof})."
    )

    # ============================================================
    # Logistics bottleneck analysis
    # ============================================================

    carrier_vs_vendor = (
        logistics_df[
            [
                "carrier_transit_days",
                "vendor_handling_days"
            ]
        ]
        .mean()
    )

    carrier_days = carrier_vs_vendor[
        "carrier_transit_days"
    ]

    vendor_days = carrier_vs_vendor[
        "vendor_handling_days"
    ]

    documents.append(
        f"Carrier transit averages {carrier_days:.1f} days, "
        f"while vendor handling averages {vendor_days:.1f} days. "
        f"Carrier transit is therefore the larger component of "
        f"logistics time."
    )

    # ============================================================
    # Logistics correlation findings
    # ============================================================

    logistics_features = [
        "is_interstate",
        "is_late",
        "vendor_handling_days",
        "carrier_transit_days",
        "distance_km",
        "mean_review_score",
        "total_freight_value"
    ]

    correlation_matrix = (
        logistics_df[logistics_features]
        .corr(method="spearman")
    )

    carrier_correlation = correlation_matrix.loc[
        "carrier_transit_days",
        "is_late"
    ]

    vendor_correlation = correlation_matrix.loc[
        "vendor_handling_days",
        "is_late"
    ]

    documents.append(
        f"Carrier transit time has a stronger association "
        f"with late deliveries (Spearman correlation = "
        f"{carrier_correlation:.2f}) than vendor handling time "
        f"(Spearman correlation = {vendor_correlation:.2f}). "
        f"This suggests that regional carrier optimization "
        f"should be considered a key logistics priority."
    )

    # ============================================================
    # Freight cost vs distance
    # ============================================================

    overall_distance_freight = (
        logistics_df[
            [
                "distance_km",
                "total_freight_value"
            ]
        ]
        .corr(method="spearman")
        .loc[
            "distance_km",
            "total_freight_value"
        ]
    )

    intrastate_distance_freight = (
        logistics_df[
            logistics_df["is_interstate"] == 0
        ][
            [
                "distance_km",
                "total_freight_value"
            ]
        ]
        .corr(method="spearman")
        .loc[
            "distance_km",
            "total_freight_value"
        ]
    )

    interstate_distance_freight = (
        logistics_df[
            logistics_df["is_interstate"] == 1
        ][
            [
                "distance_km",
                "total_freight_value"
            ]
        ]
        .corr(method="spearman")
        .loc[
            "distance_km",
            "total_freight_value"
        ]
    )

    documents.append(
        f"Freight cost has a positive association with delivery "
        f"distance. The overall Spearman correlation is "
        f"{overall_distance_freight:.2f}, with a correlation of "
        f"{intrastate_distance_freight:.2f} for intrastate shipments "
        f"and {interstate_distance_freight:.2f} for interstate "
        f"shipments. Longer-distance shipments generally tend to "
        f"have higher freight costs in both shipment types."
    )

    # ============================================================
    # Customer satisfaction
    # ============================================================

    average_review_score = (
        logistics_df["mean_review_score"].mean()
    )

    documents.append(
        f"Average customer review score is "
        f"{average_review_score:.2f} out of 5."
    )

    # ============================================================
    # RFM customer segmentation
    # ============================================================

    rfm_segments = (
        rfm["KMeans_Segment_name"]
        .value_counts()
    )

    for segment, customers in rfm_segments.items():
        documents.append(
            f"Customer segmentation: the {segment} segment "
            f"contains {customers:,} customers."
        )

    # ============================================================
    # NLP / Sentiment findings
    # ============================================================

    documents.append(
        "Negative reviews are associated with delivery and "
        "transaction complaints. Common predictive terms include "
        "péss (terrible), receb (received), vei (came), and "
        "pag (payment)."
    )

    documents.append(
        "Neutral reviews contain mixed language, including terms "
        "such as porém (however), demor (delay), and razoá "
        "(reasonable), reflecting customers acknowledging both "
        "positive and negative aspects."
    )

    documents.append(
        "Positive reviews contain praise related to speed and "
        "quality, including terms such as ant (early), excel "
        "(excellent), rápid (fast), and recom (recommend)."
    )

    documents.append(
        "Late orders have significantly lower review scores "
        "than on-time orders."
    )

    documents.append(
        "Customers tend to write longer reviews when reporting "
        "negative experiences."
    )

    documents.append(
        "Review score shows limited variation across customer "
        "segments and interstate status. Therefore, customer "
        "segment and is_interstate were excluded from the "
        "sentiment analysis."
    )

    # ============================================================
    # Sentiment distribution
    # ============================================================

    sentiment_distribution = (
        nlp_df["sentiment"]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
    )

    for sentiment, percentage in sentiment_distribution.items():
        documents.append(
            f"{percentage:.2f}% of customer reviews are "
            f"classified as {sentiment}."
        )

    return documents