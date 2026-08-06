import pytest
from src.ingest import ingestion
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

def test_duplicate_order_id(engine):
    distinct_count = pd.read_sql("SELECT COUNT(DISTINCT order_id) FROM orders",engine).iloc[0,0]
    total_count = pd.read_sql("SELECT COUNT(*) FROM orders",engine).iloc[0,0]
    assert distinct_count == total_count