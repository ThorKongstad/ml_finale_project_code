import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from joblib import dump, load

def distance_to_origin(row):
    return np.sqrt(row['dom_x']**2 + row['dom_y']**2 + row['dom_z']**2)

def distance_to_origin_string(row):
    return np.sqrt(row['dom_x']**2 + row['dom_y']**2)

data = pd.read_parquet('features.parquet', engine= 'pyarrow')

dom_coord = data[['dom_x', 'dom_y', 'dom_z']]
dom_coord = dom_coord.drop_duplicates()

dom_coord['distance_to_origin'] = dom_coord.apply(lambda row: distance_to_origin(row), axis=1)
dom_coord['dom_id'] = dom_coord['distance_to_origin'].rank(method='dense').astype(int) - 1
dom_coord['distance_to_origin_string'] = dom_coord.apply(lambda row: distance_to_origin_string(row), axis=1)
dom_coord['string_id'] = dom_coord['distance_to_origin_string'].rank(method='dense').astype(int) - 1
dom_coord = dom_coord.drop(columns=['distance_to_origin', 'distance_to_origin_string'])

scaler = StandardScaler()

scaled_values = scaler.fit_transform(dom_coord[['dom_x', 'dom_y', 'dom_z']])
scaled_dom_coord = dom_coord.copy()
scaled_dom_coord[['dom_x', 'dom_y', 'dom_z']]  = scaled_values

dump(scaler, 'std_scaler.bin', compress=True)

scaled_dom_coord['time'] = 0.0
scaled_dom_coord['x'] = scaled_dom_coord['dom_x']
scaled_dom_coord['y'] = scaled_dom_coord['dom_y']
scaled_dom_coord['z'] = scaled_dom_coord['dom_z']

scaled_dom_coord.to_parquet('graph_definition.parquet')
