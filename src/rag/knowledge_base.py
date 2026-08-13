import pandas as pd
import logging

logger = logging.getLogger(__name__)

def build_knowledge_base(logistics_df,contingency,nlp_df):

    sla_rates = (contingency[1] * 100 / contingency.sum(axis=1)).sort_values(ascending=False)
    sla_summary = ", ".join(f"{state}: {rate:.2f}%"for state, rate in sla_rates.items())
    documents = {
        'kpi':[
            f"Total revenue: R$ {logistics_df['total_price'].sum():,.2f}",
            f"Total orders: {logistics_df['order_id'].nunique():,}",
            f"Top 5 Selling Products: {','.join(logistics_df.groupby('product_category')['total_price'].sum().nlargest(5).index)}"
            ],
        'logistics': [
            f"Late delivery rate: {logistics_df['is_late'].mean():.2%}",
            f"Average delivery days: {logistics_df['delivery_days'].mean():.1f} days",
            f"Carrier transit: {logistics_df['carrier_transit_days'].mean():.1f} days",
            f"Vendor handling: {logistics_df['vendor_handling_days'].mean():.1f} days",
            f"SLA breach rates by state: {sla_summary}",
            "delivery_days has strong correlation with carrier_transit_days and vendor_handling_days",
            "Chi-square analysis (χ² = 1622.01, p < 0.001) confirms that SLA breach rates vary significantly across Brazilian states", 
            "From Co-relation analysis (0.35 vs 0.14) it is evident that carrier transit drives late deliveries,"
            "and vendor handling has minimal impact indicates regional carrier optimization should be the top logistics priority."
            ],
        'rfm': [
            "Champions: 220d recency, 2.0 freq, R$292 — highest value, need loyalty discounts.",
            "At Risk: 165d recency, 1.0 freq, R$269 — high-spend one-time buyers, need re-engagement.",
            "Lost: 428d recency, 1.0 freq, R$112 — no purchase in >1 year, win-back campaigns.",
            "Promising: 153d recency, 1.0 freq, R$47 — recent buyers, nurture to increase value."],
        'nlp': [
            f"Average review score is {nlp_df['mean_review_score'].mean():.2f}",
            f"Total reviews: {len(nlp_df):,}",
            f"Positive reviews: {(nlp_df['sentiment'] == 'positive').sum():,}",
            f"Neutral reviews: {(nlp_df['sentiment'] == 'neutral').sum():,}",
            f"Negative reviews: {(nlp_df['sentiment'] == 'negative').sum():,}"
        ]
    }
    total_docs = 0
    for section, doc in documents.items():
        count = len(doc)
        total_docs += count
        logger.info(f"{section}: {count} documents")
    flatten_documents = [doc for section in documents.values() for doc in section]

    return flatten_documents
