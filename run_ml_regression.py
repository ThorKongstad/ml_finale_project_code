import os
from typing import Any, Dict, List, Optional
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from ml_finale_project_code.dataloaders import panda_to_dataloader
from ml_finale_project_code.sim_detector_definition import app_ml_sim

import pandas as pd
from pytorch_lightning.loggers import WandbLogger
import torch
from torch.optim.adam import Adam

from graphnet.constants import EXAMPLE_DATA_DIR, EXAMPLE_OUTPUT_DIR
from graphnet.data.constants import FEATURES, TRUTH
from graphnet.models import StandardModel
from graphnet.models.detector.prometheus import Prometheus
from graphnet.models.gnn import DynEdge
from graphnet.models.graphs import KNNGraph
from graphnet.models.task.reconstruction import EnergyReconstruction
from graphnet.training.callbacks import PiecewiseLinearLR
from graphnet.training.loss_functions import LogCoshLoss
from graphnet.training.utils import make_train_validation_dataloader
from graphnet.utilities.argparse import ArgumentParser
from graphnet.utilities.logging import Logger





def main(
        Training_parquet_dir: str,
        Truth_parquet_dir: str,
        target: str,
        saved_model: Optional[str] = None,
        gpus: Optional[List[int]] = None,
        max_epochs: int = 1,
        early_stopping_patience: int = 100,
        batch_size: int = 16,
        num_workers: int = 1,
        ):

    logger = Logger()

    training_pd = pd.read_parquet(Training_parquet_dir)
    truth_pd = pd.read_parquet(Truth_parquet_dir)

    logger.info(f"features: {'; '.join(training_pd.columns)}")
    logger.info(f"truth: {'; '.join(truth_pd.columns)}")

    config: Dict[str, Any] = {
        "path": Training_parquet_dir,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "target": target,
        "early_stopping_patience": early_stopping_patience,
        "fit": {
            "gpus": gpus,
            "max_epochs": max_epochs,
        },
    }

    graph_definition = KNNGraph(detector=app_ml_sim())

    (
        training_dataloader,
    ) = panda_to_dataloader(db=truth_pd, truth_pd=truth_pd, graph_definition=graph_definition,
                            features=training_pd.columns.to_list(), truth=truth_pd.columns.to_list(), batch_size=config["batch_size"],
                            num_workers=config["num_workers"])

    # Building model

    backbone = DynEdge(
        nb_inputs=graph_definition.nb_outputs,
        global_pooling_schemes=["min", "max", "mean", "sum"],
    )

    task = EnergyReconstruction(
        hidden_size=backbone.nb_outputs,
        target_labels=config["target"],
        loss_function=LogCoshLoss(),
        transform_prediction_and_target=lambda x: torch.log10(x),
        transform_inference=lambda x: torch.pow(10, x),
    )

    model = StandardModel(
        graph_definition=graph_definition,
        backbone=backbone,
        tasks=[task],
        optimizer_class=Adam,
        optimizer_kwargs={"lr": 1e-03, "eps": 1e-03},
        scheduler_class=PiecewiseLinearLR,
        scheduler_kwargs={
            "milestones": [
                0,
                len(training_dataloader) / 2,
                len(training_dataloader) * config["fit"]["max_epochs"],
            ],
            "factors": [1e-2, 1, 1e-02],
        },
        scheduler_config={
            "interval": "step",
        },
    )

    # Training model
    model.fit(
        training_dataloader,
#        validation_dataloader,
        early_stopping_patience=config["early_stopping_patience"],
        logger= None,
        **config["fit"],
    )


    model.save(f"./model.pth")
    model.save_state_dict(f"./state_dict.pth")
    model.save_config(f"./model_config.yml")



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('Training_parquet_dir')
    #parser.add_argument('Validation_parquet_dir')
    parser.add_argument('Truth_parquet_dir')
    parser.add_argument('--saved_model', '-model', default=None, type=str)
    parser.add_argument(
        "--target",
        help=(
            "Name of feature to use as regression target (default: "
            "%(default)s)"
        ),
        default="energy",
    )

    parser.with_standard_arguments(
        "gpus",
        ("max-epochs", 1),
        "early-stopping-patience",
        ("batch-size", 16),
        "num-workers",
    )

    args, unknown = parser.parse_known_args()


    main(**args.__dict__)
    #main(Training_parquet_dir=args.Training_parquet_dir,
    #     Truth_parquet_dir=args.Truth_parquet_dir,
    #     target=args.target,
    #     gpus=args.gpus,
    #     )
