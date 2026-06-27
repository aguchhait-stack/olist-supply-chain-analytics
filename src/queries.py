from src.ingest import engine
import pandas as pd

# Top 10 revenue catagories
def get_top_revenue_categories(engine):
    query = """
        WITH order_items_CTE AS (
            SELECT *, order_item_id * (price + freight_value) AS total_order_value_per_order_id
            FROM order_items
            ),
        total_order_value_CTE AS (
            SELECT product_id, SUM(total_order_value_per_order_id) AS total_order_value
            FROM order_items_CTE
            GROUP BY product_id
        )
        SELECT COALESCE(product_category_name_english,product_category_name,'unknown') AS  product_category,
               SUM(total_order_value) AS category_wise_total_order_value
        FROM products

        LEFT JOIN product_category_name_translation USING (product_category_name)
        LEFT JOIN total_order_value_CTE USING (product_id)

        GROUP BY COALESCE(product_category_name_english,product_category_name,'unknown')
        ORDER BY category_wise_total_order_value DESC;
    """
    result_sql = pd.read_sql(query,engine)
    result_sql["category_wise_total_order_value"] = result_sql["category_wise_total_order_value"]* 0.15 # BRL to GBP (June 2026)
    data = result_sql.head(10).sort_values(by= 'category_wise_total_order_value')
    return data

if __name__=="__main__":
    get_top_revenue_categories(engine)




