"""Episode logger in the real-world data format, usable outside ROS.

Writes one hdf5 file per episode under ``datasets/real/<dataset_name>/raw/<date>/``
with the same layout as the ROS controller's logger
(``go2_ros2_ws/.../episode_logger_hdf5.py``), so the files can be aggregated with
``scripts/aggregate_realworld_data.py``. Extra per-step scalars (for example the
simulator reward in sim-to-sim experiments) can be logged alongside.
"""

import os
import time
from datetime import datetime

import h5py
import numpy as np

from simdist.data import REAL_DATA_KEYS
from simdist.utils import paths

TIME_KEY = REAL_DATA_KEYS["time"]
PROPRIO_KEY = REAL_DATA_KEYS["proprio_obs"]
EXTERO_KEY = REAL_DATA_KEYS["extero_obs"]
ACTS_KEY = REAL_DATA_KEYS["actions"]
CMDS_KEY = REAL_DATA_KEYS["commands"]


class HDF5EpisodeLogger:
    def __init__(self, dataset_name: str, extra_keys: tuple[str, ...] = ()):
        self.dataset_name = dataset_name
        self.extra_keys = tuple(extra_keys)
        self.file = None
        self.path = None
        self.datasets = {}
        self.step = 0
        self.num_episodes = 0

    def open(self) -> str:
        """Start a new episode file. Closes the previous one if still open."""
        if self.file is not None:
            self.close()
        date = datetime.now().strftime("%Y-%m-%d")
        t = datetime.now().strftime("%H-%M-%S-%f")[:-3]
        raw_dir = paths.get_real_raw_data_dir(self.dataset_name)
        self.path = os.path.join(raw_dir, date, f"{t}.hdf5")
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self.file = h5py.File(self.path, "w")

        self.datasets = {
            TIME_KEY: self.file.create_dataset(
                TIME_KEY, shape=(0,), maxshape=(None,), dtype="int64", chunks=True
            )
        }
        for key in (PROPRIO_KEY, EXTERO_KEY, ACTS_KEY, CMDS_KEY):
            self.datasets[key] = self.file.create_dataset(
                key, shape=(0, 0), maxshape=(None, None), dtype="float32", chunks=True
            )
        for key in self.extra_keys:
            self.datasets[key] = self.file.create_dataset(
                key, shape=(0,), maxshape=(None,), dtype="float32", chunks=True
            )
        self.step = 0
        return self.path

    def write(
        self,
        proprio_obs: np.ndarray,
        extero_obs: np.ndarray,
        action: np.ndarray,
        command: np.ndarray,
        extras: dict[str, float] | None = None,
        timestamp_ns: int | None = None,
    ) -> None:
        if self.file is None:
            return
        if self.step == 0:
            self.datasets[PROPRIO_KEY].resize((0, np.size(proprio_obs)))
            self.datasets[EXTERO_KEY].resize((0, np.size(extero_obs)))
            self.datasets[ACTS_KEY].resize((0, np.size(action)))
            self.datasets[CMDS_KEY].resize((0, np.size(command)))

        n = self.step + 1
        for key, dset in self.datasets.items():
            if dset.ndim == 1:
                dset.resize((n,))
            else:
                dset.resize((n, dset.shape[1]))

        self.datasets[TIME_KEY][self.step] = (
            time.time_ns() if timestamp_ns is None else int(timestamp_ns)
        )
        self.datasets[PROPRIO_KEY][self.step] = np.asarray(proprio_obs, np.float32).ravel()
        self.datasets[EXTERO_KEY][self.step] = np.asarray(extero_obs, np.float32).ravel()
        self.datasets[ACTS_KEY][self.step] = np.asarray(action, np.float32).ravel()
        self.datasets[CMDS_KEY][self.step] = np.asarray(command, np.float32).ravel()
        for key in self.extra_keys:
            self.datasets[key][self.step] = np.float32((extras or {}).get(key, 0.0))
        self.step += 1

    def close(self) -> int:
        """Finish the current episode; empty episodes are deleted. Returns its length."""
        if self.file is None:
            return 0
        steps = self.step
        self.file.flush()
        self.file.close()
        self.file = None
        if steps == 0:
            os.remove(self.path)
        else:
            self.num_episodes += 1
        self.datasets = {}
        self.step = 0
        return steps
