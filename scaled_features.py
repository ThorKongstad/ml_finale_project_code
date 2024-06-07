import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from joblib import dump, load

scaler = load('std_scaler.bin')

data = pd.read_parquet('features.parquet', engine= 'pyarrow')
dom_coord = pd.read_parquet('graph_definition.parquet', engine = 'pyarrow')

merged_dataset = data.merge(dom_coord, on=['dom_x', 'dom_y', 'dom_z'], how='left')
merged_dataset = merged_dataset.drop(columns = ['string_id'])

data_scaled_values = scaler.fit_transform(merged_dataset[['dom_x', 'dom_y', 'dom_z', 'dom_time', 'charge']])
scaled_data = merged_dataset.copy()
scaled_data[['dom_x', 'dom_y', 'dom_z', 'dom_time', 'charge']] = data_scaled_values

scaled_data.to_parquet('scaled_features.parquet')