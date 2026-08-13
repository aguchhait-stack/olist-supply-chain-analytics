
from src.data.ingest import engine
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

def _build_master_dataframe(engine) -> pd.DataFrame:
    """Create the transaction-level master dataframe."""
    MASTER_QUERY = """

    SELECT 
    --  Orders
        o.order_id, o.order_status, o.order_purchase_timestamp, o.order_approved_at,
        o.order_delivered_carrier_date, o.order_delivered_customer_date, o.order_estimated_delivery_date,
        
    --  Customers
    c.customer_unique_id, c.customer_zip_code_prefix, c.customer_city, c.customer_state, 

    -- order_items
    oi.order_item_id, oi.price, oi.freight_value, oi.seller_id, oi.product_id,

    -- products & translation
    COALESCE(pt.product_category_name_english,p.product_category_name,'unknown') AS product_category,

    -- sellers
    s.seller_state, s.seller_city, s.seller_zip_code_prefix,

    -- order_payments
    op.payment_type, op.payment_value,

    -- order_reviews
    r.review_score, r.review_comment_message

    FROM orders o
    INNER JOIN customers c USING(customer_id)
    LEFT JOIN order_items oi USING(order_id)
    LEFT JOIN sellers s USING(seller_id)
    LEFT JOIN products p USING(product_id)
    LEFT JOIN product_category_name_translation pt USING(product_category_name) 
    LEFT JOIN order_payments op USING(order_id)
    LEFT JOIN order_reviews r  USING(order_id)

    """
    master_df = pd.read_sql(MASTER_QUERY,engine)

    # Converting all datetime columns to pandas datetime64 dtype
    for col in master_df.columns[master_df.columns.str.contains("date|timestamp|approved_at")]:
            master_df[col] = pd.to_datetime(master_df[col],errors="coerce")
    logger.info(f"Master dataframe: {master_df.shape[0]:,} rows, {master_df.shape[1]} columns")
    return master_df

def _aggregate_orders(master_df: pd.DataFrame) -> pd.DataFrame:
    """
    - Convert the transaction-level dataset into one row per order.
    - Analysis limited to **delivered orders only** (97% of all orders).
    - Aggregation at `order_id` level (before dedup).
    """
    # Logistics DataFrame
    logistics_df = master_df.copy()
    aggregation_map = {

    "price":                  ("total_price","sum"),
    "freight_value":          ("total_freight_value","sum"),
    "payment_value":          ("total_payment_value","sum"),
    "review_score" :          ("mean_review_score","mean"),

    # Aggregation with first occurance for review comment with null value check
    "review_comment_message": ("review_comment_message", lambda x: x[x.notna()].iloc[0] if x.notna().any() else np.nan),
    "product_id":             ("product_id","first"),
    "seller_id":              ("seller_id","first"),
    "seller_zip_code_prefix": ("seller_zip_code_prefix", lambda x: x[x.notna()].iloc[0] if x.notna().any() else np.nan),

    # Aggregation with most occurance for string columns
    "product_category":       ("product_category", lambda x: x[x.notna()].mode()[0] if x.notna().any() else np.nan),
    "seller_state":           ("seller_state",lambda x: x[x.notna()].mode()[0] if x.notna().any() else np.nan),
    "seller_city":            ("seller_city",lambda x: x[x.notna()].mode()[0] if x.notna().any() else np.nan),
    "payment_type":           ("payment_type",lambda x: x[x.notna()].mode()[0] if x.notna().any() else np.nan)
    }
    # Aggregation without collapsing any rows uisng the logic in aggregation_map
    for col, (new_col,func) in aggregation_map.items():
        logistics_df[new_col] = logistics_df.groupby('order_id')[col].transform(func)

    # droping non-aggregation columns
    col_to_drop = ['price', 'freight_value', 'payment_value', 'review_score', 'order_item_id']
    logistics_df = logistics_df.drop(columns = col_to_drop)

    # Fitering null dates
    logistics_df = logistics_df.dropna(subset='order_delivered_customer_date').reset_index(drop=True)

    return logistics_df 

def _add_order_level_flags(logistics_df: pd.DataFrame) -> pd.DataFrame:
    
    ## Geography, Same order_id has multiple pair, using max occurance for intertstate flag
    logistics_df['is_interstate'] = (logistics_df['customer_state'] != logistics_df['seller_state']).astype(int)
    logistics_df['is_interstate'] = logistics_df.groupby('order_id')['is_interstate'].transform('max')
    return logistics_df 

def _deduplicate_orders(logistics_df: pd.DataFrame) -> pd.DataFrame: 
    logistics_df = logistics_df.drop_duplicates(subset='order_id').reset_index(drop=True)
    return logistics_df

def _add_order_features(logistics_df: pd.DataFrame) -> pd.DataFrame:

    ## SLA & Delivery
    logistics_df['is_late'] = (logistics_df['order_delivered_customer_date'] >logistics_df['order_estimated_delivery_date']).astype(int)
    logistics_df['delivery_days'] = (logistics_df['order_delivered_customer_date'] -logistics_df['order_purchase_timestamp']).dt.days
    logistics_df["vendor_handling_days"] = (logistics_df["order_delivered_carrier_date"]-logistics_df["order_purchase_timestamp"]).dt.days
    logistics_df["carrier_transit_days"] = (logistics_df["order_delivered_customer_date"]-logistics_df["order_delivered_carrier_date"]).dt.days
    return logistics_df 

def _add_geolocation(logistics_df: pd.DataFrame, engine=engine) -> pd.DataFrame: 

    # Aggregate geolocation table
    GEOLOCATION_QUERY = """
    SELECT geolocation_zip_code_prefix, AVG(geolocation_lat) geolocation_lat, AVG(geolocation_lng) geolocation_lng
    FROM geolocation
    GROUP BY geolocation_zip_code_prefix
    """
    grouped_geolocation = pd.read_sql(GEOLOCATION_QUERY,engine)

    # joining customer, seller location with geolocation table
    logistics_df = logistics_df.merge(grouped_geolocation,left_on = 'customer_zip_code_prefix',right_on = 'geolocation_zip_code_prefix',how= 'left').\
                rename(columns = {'geolocation_lat': 'cust_latitude', 'geolocation_lng': 'cust_longitude'})
    logistics_df = logistics_df.merge(grouped_geolocation,left_on = 'seller_zip_code_prefix',right_on = 'geolocation_zip_code_prefix',how= 'left').\
                rename(columns = {'geolocation_lat': 'seller_latitude', 'geolocation_lng': 'seller_longitude'})

    # Dropping 'zip_code_prefix' columns  after merge
    logistics_df = logistics_df.drop(columns=['customer_zip_code_prefix','seller_zip_code_prefix','geolocation_zip_code_prefix_x',\
                                              'geolocation_zip_code_prefix_y']).reset_index(drop=True)
    return logistics_df

def _add_haversine_distance(logistics_df: pd.DataFrame) -> pd.DataFrame:

    lat1 = np.radians(logistics_df['cust_latitude'])
    lat2 = np.radians(logistics_df['seller_latitude'])
    lon1 = np.radians(logistics_df['cust_longitude'])
    lon2 = np.radians(logistics_df['seller_longitude'])
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # Haversign formula
    logistics_df['distance_km'] = 2*6371*np.arcsin(np.sqrt(np.sin(dlat/2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2))

    # Filling missing values with median
    logistics_df['distance_km'] = logistics_df['distance_km'].fillna(logistics_df['distance_km'].median())
    return logistics_df

def build_logistics_dataframe(engine=engine) -> pd.DataFrame:
    """
    Build the canonical logistics dataset.

    Pipeline
    --------
    1. Build transaction-level master dataframe.
    2. Aggregate records to one row per order.
    3. Create multi-row features.
    4. Deduplicate orders.
    5. Create order-level features.
    6. Enrich with geolocation.
    7. Calculate customer–seller distance.

    Returns
    -------
    pd.DataFrame
        One row per delivered order with engineered logistics features.
    """

    master_df =    _build_master_dataframe(engine)
    logistics_df = _aggregate_orders(master_df)
    logistics_df = _add_order_level_flags(logistics_df)
    logistics_df = _deduplicate_orders(logistics_df)
    logistics_df = _add_order_features(logistics_df)
    logistics_df = _add_geolocation(logistics_df, engine)
    logistics_df = _add_haversine_distance(logistics_df)
    logger.info(f"✅ Logistics dataframe: {logistics_df.shape[0]:,} rows, {logistics_df.shape[1]} columns")
    return logistics_df