import pandas as pd

def _find_aggregation_columns(master_df: pd.DataFrame) -> dict[str, object]:
    """
    Identify columns that require aggregation before creating  before constructing
    the order-level logistics dataset.
    """
    dedup_dict = {}
    for col in master_df.columns:
        if col not in ['order_id','order_item_id'] and (master_df.groupby('order_id')[col].nunique()>1).any():
            print(f'Need aggregation: {col}, {master_df[col].dtype}')

    return dedup_dict 

def validate_nlp_df(nlp_df: pd.DataFrame) -> None:
    if nlp_df.empty:
        raise ValueError("NLP dataframe is empty — no reviewed orders found")
    dupes = nlp_df.duplicated().sum()
    if dupes:
        raise ValueError(f"{dupes} duplicate rows in nlp_df")