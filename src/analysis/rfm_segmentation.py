import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import logging
logger = logging.getLogger(__name__)
RANDOM_STATE = 42

def _build_rfm_table(logistics_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create the customer-level RFM table.
    """
    snap_date = max(logistics_df["order_purchase_timestamp"])+ pd.Timedelta(1,'D')
    rfm = logistics_df.groupby('customer_unique_id').agg(
        recency = ('order_purchase_timestamp',lambda x: (snap_date-x.max()).days),
        frequency = ('order_id','nunique'),
        monetary  = ('total_price','sum')
        ).reset_index()
    return rfm

def _fit_kmeans(rfm: pd.DataFrame,n_clusters:int = 4):
    """
    Train a KMeans model and assign customer segments.
    """
    X = rfm[['recency', 'frequency', 'monetary']].copy()
    X['monetary']  = np.log1p(X['monetary'])

    # train-test split with 80% train set, random state for reproducibility
    X_train,X_test = train_test_split(X, test_size=0.2,random_state=RANDOM_STATE)

    # Standardization required for KMeans Euclidean distance,
    # Fit only on train set to ensure no data leakage.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Transform only on test set to prevent data leakage
    X_test_scaled = scaler.transform(X_test)

    # Final model run using the cluster, k=4
    kmeans = KMeans(n_clusters = n_clusters, random_state=RANDOM_STATE)
    kmeans.fit(X_train_scaled)

    # Predicted labels on train dataset
    train_labels = kmeans.labels_ 

    # Predicted labels on train dataset
    test_labels  = kmeans.predict(X_test_scaled)

    # assign labels back to rfm tables with right index in new column
    rfm.loc[X_train.index,'KMeans_Segment'] = train_labels
    rfm.loc[X_test.index,'KMeans_Segment']  = test_labels
    rfm['KMeans_Segment'] = rfm['KMeans_Segment'].astype(int)

    return rfm, X_train_scaled, X_test_scaled, kmeans


def _merge_customer_segments(logistics_df: pd.DataFrame,rfm: pd.DataFrame) -> pd.DataFrame:
    """
    Merge customer segment labels into the logistics dataframe.
    """
    segment_map = {0: "Lost", 1: "At Risk", 2: "Champions", 3: "Promising"}

    if rfm['KMeans_Segment'].nunique() != len(segment_map):
        print('Update segment mapping.')
        return logistics_df

    # Mapping the Segemnt according to standard rfm interpretation
    rfm['KMeans_Segment_name'] = rfm['KMeans_Segment'].map(segment_map)
    # Bringing back rfm values to logistics_df
    hashmap = rfm.set_index("customer_unique_id")["KMeans_Segment_name"]
    logistics_df["customer_segment"] = logistics_df["customer_unique_id"].map(hashmap)
    return logistics_df

def build_customer_segmentation(logistics_df: pd.DataFrame, n_clusters: int = 4):
    """
    Build customer segmentation using RFM and KMeans.

    Pipeline
    --------
    1. Create the RFM table.
    2. Train the KMeans model.
    3. Assign customer segments.
    4. Merge segments into the logistics dataframe.

    Returns
    -------
    tuple
        RFM table, logistics dataframe, scaled train/test features, and trained KMeans model.
    """
    rfm = _build_rfm_table(logistics_df)
    rfm, X_train_scaled, X_test_scaled,kmeans  = _fit_kmeans(rfm,n_clusters=n_clusters)
    logistics_df = _merge_customer_segments(logistics_df, rfm)
    logger.info("Segmentation complete: %d segments created",rfm["KMeans_Segment_name"].nunique())
    return logistics_df, rfm, X_train_scaled, X_test_scaled, kmeans 


