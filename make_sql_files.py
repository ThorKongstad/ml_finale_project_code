import pandas as pd
import sqlite3
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split

features = pd.read_parquet('features.parquet', engine = 'pyarrow')
truth = pd.read_parquet('truth.parquet', engine = 'pyarrow')

truth_train, truth_test = train_test_split(
    truth,
    test_size=0.925,
    random_state=42,
    shuffle=True
)

truth_train_events = truth_train['event_no']
train_features = features.query('event_no.isin(@truth_train_events)')
# test_features = features.query('~event_no.isin(@truth_train_events)')

# Create a SQLAlchemy engine
engine = create_engine('sqlite:///mid_train_icecube_events.db')

# Save DataFrames to the database as tables
train_features.to_sql('total', engine, if_exists='replace', index=False)
truth_train.to_sql('mc_truth', engine, if_exists='replace', index=False)

# Dispose of the engine (close connection)
engine.dispose()