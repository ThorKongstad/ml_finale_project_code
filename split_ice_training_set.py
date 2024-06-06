import argparse
import os

import pandas as pd
import sklearn as sk


def main(truth_table: str, measurements_table: str, split_ratio: float):

    truth_table_pd = pd.read_parquet(truth_table)
    measurements_table_pd = pd.read_parquet(measurements_table)

    truth_train, truth_val = sk.model_selection.train_test_split(
        truth_table_pd,
        test_size=split_ratio,
        random_state=42,
        shuffle=True
    )

    truth_train_events = truth_train['event_no']

    measurements_train_table = measurements_table_pd.query('event_no.isin(@truth_train_events)')
    measurements_val_table = measurements_table_pd.query('~event_no.isin(@truth_train_events)')

    truth_train.to_parquet('./' + os.path.basename(truth_table).split('.')[0] + '_train.parquet')
    truth_val.to_parquet('./' + os.path.basename(truth_table).split('.')[0] + '_validation.parquet')

    measurements_train_table.to_parquet('./' + os.path.basename(measurements_table).split('.')[0] + '_train.parquet')
    measurements_val_table.to_parquet('./' + os.path.basename(measurements_table).split('.')[0] + '_validation.parquet')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('Truth_events')
    parser.add_argument('measurement_table')
    parser.add_argument('--split_ratio', '-split', default=0.3, type=float)
