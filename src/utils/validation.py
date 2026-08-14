import pandas as pd
import logging

logger = logging.getLogger(__name__)

def validate_dataframe(df: pd.DataFrame)-> None:
    """
    Validate a dataframe for duplicate rows and missing values.
    """
    print("-" * 30)
    print("Null & Duplicate Validation")
    print("-" * 30)
    dupes = df.duplicated().sum()
    nulls = df.isna().sum()[df.isna().sum() > 0]
    if dupes > 0:
        logger.warning(f"⚠️ Duplicate rows: {dupes:,}")
    else:
            logger.info(f"✅ No duplicates")
    if len(nulls) > 0:
            logger.warning("⚠️ Missing values found:")
            for col, count in nulls.items():
                logger.warning(f"   {col}: {count:,}")
    else:
        logger.info("✅ No missing values found")
def find_aggregation_columns(master_df: pd.DataFrame) -> dict:
    """
    Identify columns that require aggregation before creating  before constructing
    the order-level logistics dataset.
    """
    print("-" * 30)
    print("Aggregation needed at transaction level")
    print("-" * 30)
    agg_cols = {}
    for col in master_df.columns:
        if col not in ['order_id','order_item_id'] and (master_df.groupby('order_id')[col].nunique()>1).any():
            logger.info(f"Need aggregation: {col}, {master_df[col].dtype}")
            agg_cols[col] = master_df[col].dtype

    return agg_cols