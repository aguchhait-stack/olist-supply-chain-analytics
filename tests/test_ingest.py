import pytest
from src.data.ingest import ingestion
from src.data.logistics_data import build_logistics_dataframe
import pandas as pd


@pytest.fixture(scope='session')
def engine():
    return ingestion()

def test_engine(engine):
    assert engine is not None

def test_tables(engine):
    tables = pd.read_sql("SELECT name FROM sqlite_master where type = 'table'",engine)
    expected_tables = ['customers', 'geolocation','order_items', 'order_payments', 'order_reviews',
                       'orders', 'product_category_name_translation','products','sellers']
    assert len(tables) == 9
    assert set(expected_tables) == set(tables['name'])

def test_logistics_df(engine):
    logistics_df = build_logistics_dataframe(engine)
    order_counts = logistics_df.groupby("order_id").size()

    # Duplicate orders
    assert (order_counts > 1).sum() == 0 
