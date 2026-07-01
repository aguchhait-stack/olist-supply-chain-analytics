from src.ingest import engine
import pandas as pd

def load_joined_data(engine) -> pd.DataFrame:

    """
    Joins 8 tables of the Olist e-commerce database to build a unified dataset 
    covering marketing, logistics, financial, and sentiment analysis.
    
    Args:
        engine: Database connection engine.

    Returns:
        pd.DataFrame: A row-level DataFrame containing 23 key analytical columns.
    """

    query = """
    SELECT 
        -- CUSTOMERS 
        c.*,

        -- ORDERS
        o.order_id, o.order_status, o.order_purchase_timestamp, o.order_approved_at, 
        o.order_delivered_carrier_date, o.order_delivered_customer_date, o.order_estimated_delivery_date,

        -- ORDER_ITEMS
        oi.order_item_id, oi.price, oi.freight_value, oi.seller_id,

        -- PAYMENTS
        op.payment_type,

        -- PRODUCTS & TRANSLATION
        p.product_id, COALESCE(t.product_category_name_english, p.product_category_name,'unknown') AS product_category,

        -- SELLERS
        s.seller_zip_code_prefix, s.seller_city, s.seller_state,

        -- REVIEWS
        r.review_score, r.review_comment_message

    From Customers c
    INNER JOIN Orders o USING(customer_id)
    LEFT JOIN Order_items oi USING(order_id)
    LEFT JOIN products p USING(product_id)
    LEFT JOIN Product_category_name_translation t USING(product_category_name)
    LEFT JOIN Sellers s  USING(seller_id)
    LEFT JOIN Order_payments op USING (order_id)
    LEFT JOIN Order_reviews r USING(order_id)
    """
    joined_df = pd.read_sql(query,engine)
    date_col = ['order_purchase_timestamp','order_delivered_carrier_date','order_delivered_customer_date','order_estimated_delivery_date']
    for col in date_col:
        joined_df[col] = pd.to_datetime(joined_df[col],errors='coerce')
    return joined_df

if __name__ == "__main__":
    df = load_joined_data(engine)

