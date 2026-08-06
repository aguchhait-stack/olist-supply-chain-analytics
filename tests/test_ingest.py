import pytest
from sqlalchemy import create_engine
from src.ingest import ingestion
import pandas as pd
import os


@pytest.fixture
def engine():
    return create_engine('sqlite:///data/ecommerce.db')

def test_ingestion():
    engine = ingestion()
    assert engine is not None

def test_ingestion_failure():
    with pytest.raises(Exception):
        ingestion(path="/invalid/path")


@pytest.mark.skipif(not os.path.exists('data/ecommerce.db'),reason = 'Run ingestion() first.')
def test_tables(engine):
    tables = pd.read_sql("SELECT name FROM sqlite_master where type = 'table'",engine)
    expected_tables = ['customers', 'geolocation','order_items', 'order_payments', 'order_reviews',
                       'orders', 'product_category_name_translation','products','sellers']
    assert len(tables) == 9
    assert set(expected_tables) == set(tables['name'])