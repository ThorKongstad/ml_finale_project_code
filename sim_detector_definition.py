import os
from typing import Dict, Callable

from graphnet.models.detector.detector import Detector
import torch


class app_ml_sim(Detector):
    """Detector class for the simulated setup"""

    geometry_table_path = os.path.join(os.path.dirname(__file__), 'graph_definition.parquet')

    xyz = ['dom_x', 'dom_y', 'dom_z']
    sensor_id_column = 'dom_id'
    string_id_column = 'string_id'

    def feature_map(self) -> Dict[str, Callable]:
        """Map standardization functions to each dimension."""
        feature_map = {
            "dom_x": self._sensor_pos_xyz,
            "dom_y": self._sensor_pos_xyz,
            "dom_z": self._sensor_pos_xyz,
            "time": self._t,
            "charge": self._charge,

        }
        return feature_map

    def _sensor_pos_xyz(self, x: torch.tensor) -> torch.tensor:
        return x

    def _t(self, x: torch.tensor) -> torch.tensor:
        return x

    def _charge(self, x: torch.tensor) -> torch.tensor:
        return x


class app_ml_sim_scalling(Detector):
    """Detector class for the simulated setup"""

    geometry_table_path = os.path.join(os.path.dirname(__file__), 'graph_definition.parquet')

    xyz = ['dom_x', 'dom_y', 'dom_z']
    sensor_id_column = 'dom_id'
    string_id_column = 'string_id'

    def feature_map(self) -> Dict[str, Callable]:
        """Map standardization functions to each dimension."""
        feature_map = {
            "dom_x": self._dom_xyz,
            "dom_y": self._dom_xyz,
            "dom_z": self._dom_xyz,
            "dom_time": self._dom_time,
            "charge": self._charge,

        }
        return feature_map


    def _dom_xyz(self, x: torch.tensor) -> torch.tensor:
        return x / 500.0

    def _dom_time(self, x: torch.tensor) -> torch.tensor:
        return (x - 1.0e04) / 3.0e4

    def _charge(self, x: torch.tensor) -> torch.tensor:
        return torch.log10(x)
