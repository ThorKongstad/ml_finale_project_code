import sqlite3
import tempfile
#from collections import OrderedDict
import os
from typing import Dict, List, Optional, Tuple, Union, Callable

import numpy as np
import pandas as pd
#from pytorch_lightning import Trainer
#from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
#from torch_geometric.data import Batch, Data

#from graphnet.data.dataset import Dataset
from graphnet.data.dataset import SQLiteDataset
#from graphnet.data.dataset import ParquetDataset
#from graphnet.models import Model
#from graphnet.utilities.logging import Logger
from graphnet.models.graphs import GraphDefinition
from graphnet.training.utils import collate_fn


def make_dataloader(
    db: str,
    pulsemaps: Union[str, List[str]],
    graph_definition: GraphDefinition,
    features: List[str],
    truth: List[str],
    *,
    batch_size: int,
    shuffle: bool,
    selection: Optional[List[int]] = None,
    num_workers: int = 10,
    persistent_workers: bool = True,
    node_truth: List[str] = None,
    truth_table: str = "truth",
    node_truth_table: Optional[str] = None,
    string_selection: List[int] = None,
    loss_weight_table: Optional[str] = None,
    loss_weight_column: Optional[str] = None,
    index_column: str = "event_no",
    labels: Optional[Dict[str, Callable]] = None,
) -> DataLoader:
    """Construct `DataLoader` instance."""
    # Check(s)
    if isinstance(pulsemaps, str):
        pulsemaps = [pulsemaps]

    dataset = SQLiteDataset(
        path=db,
        pulsemaps=pulsemaps,
        features=features,
        truth=truth,
        selection=selection,
        node_truth=node_truth,
        truth_table=truth_table,
        node_truth_table=node_truth_table,
        string_selection=string_selection,
        loss_weight_table=loss_weight_table,
        loss_weight_column=loss_weight_column,
        index_column=index_column,
        graph_definition=graph_definition,
    )

    # adds custom labels to dataset
    if isinstance(labels, dict):
        for label in labels.keys():
            dataset.add_label(key=label, fn=labels[label])

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        persistent_workers=persistent_workers,
        prefetch_factor=2,
    )

    return dataloader


def panda_to_dataloader(
    db: pd.DataFrame,
    truth_pd: pd.DataFrame,
    graph_definition: GraphDefinition,
    #pulsemaps: Union[str, List[str]],
    features: List[str],
    truth: List[str],
    *,
    batch_size: int,
    seed: int = 42,
    num_workers: int = 10,
    persistent_workers: bool = True,
    node_truth: Optional[str] = None,
#    truth_table: str = "truth",
    node_truth_table: Optional[str] = None,
    string_selection: Optional[List[int]] = None,
    loss_weight_column: Optional[str] = None,
    loss_weight_table: Optional[str] = None,
    index_column: str = "event_no",
    labels: Optional[Dict[str, Callable]] = None,
) -> DataLoader:
    """Construct train and test `DataLoader` instances."""
    # Reproducibility
    rng = np.random.default_rng(seed=seed)
    # Checks(s)
    #if isinstance(pulsemaps, str):
        #pulsemaps = [pulsemaps]

    # SAVE panda as sql
#    tempdir = tempfile.TemporaryDirectory(prefix='tmp_sqlites')
    #con = sqlite3.connect(sql_path:=f'{tempdir.name}/sql_{os.getpid()}.db')
    con = sqlite3.connect(sql_path:=f'graph_sql_tmp.db')
    db.to_sql(f'pulsemap', con, if_exists='replace', index=False)
    truth_pd.to_sql('truth', con, if_exists='replace', index=False)
    con.close()

    print('Lookk hereee:  ', os.path.exists(sql_path))

    # Create DataLoaders
    common_kwargs = dict(
        db=sql_path,
        pulsemaps=['pulsemap'],
        features=features,
        truth=truth,
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        node_truth=node_truth,
        truth_table='truth',
        node_truth_table=node_truth_table,
        string_selection=string_selection,
        loss_weight_column=loss_weight_column,
        loss_weight_table=loss_weight_table,
        index_column=index_column,
        labels=labels,
        graph_definition=graph_definition,
    )

    dataloader = make_dataloader(
        shuffle=True,
        #selection=training_selection,
        **common_kwargs,  # type: ignore[arg-type]
    )

    return dataloader
