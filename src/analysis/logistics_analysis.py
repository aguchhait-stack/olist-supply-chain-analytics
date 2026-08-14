import pandas as pd
from scipy.stats import chi2_contingency
import logging
logger = logging.getLogger(__name__)
ALPHA = 0.05

def analyze_sla_by_state(logistics_df: pd.DataFrame) -> tuple:
    """
    Test whether SLA breach rates vary significantly across customer states.
    """
    contingency = pd.crosstab(logistics_df["customer_state"],logistics_df["is_late"])

    # Chi-square test
    chi2, p, dof, expected = chi2_contingency(contingency)
    logger.info(f"Chi2: {chi2:.2f}, p-value: {p:.2e}, alpha: {ALPHA}")

    if p < ALPHA:
        logger.info("SLA breach rate varies significantly across states")
    else:
        logger.info("SLA breach rate is uniform across all Brazilian states")

    return contingency, chi2, p, dof, expected

