import random

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import Tensor
from torch.utils.data import Dataset


def _numpy(X: np.ndarray) -> Tensor:
    return torch.from_numpy(X).float()


class TelenorDataset(Dataset):

    split_map = {
        "train": 0,
        "val": 1,
        "test": 2,
    }

    binary_features = [
        "mcdr_nom_d",
        "mcdr_nom_s",
        "msdr_nom_d",
        "msdr_nom_s",
        "ho_nom",
    ]

    def __init__(
        self,
        sector_id: str | None = None,
        n_sectors: int | None = None,
        seq_len: int = 512,
        pred_len: int = 0,
        split="train",
        drop_binary_features: bool = True,
        split_size: tuple[float, float, float] = (0.6, 0.2, 0.2),
        scale: bool = True,
        is_pretrain: bool = False,
    ):
        if (sector_id is None and n_sectors is None) or (
            sector_id is not None and n_sectors is not None
        ):
            raise ValueError(
                "Exactly one of sector_id or n_sectors must be defined, but not both."
            )

        if split not in [
            "train",
            "val",
            "test",
        ]:
            raise ValueError(
                f"task_type must be one of 'train', 'val', 'test'. Got '{split}'"
            )

        if np.sum(split_size) != 1.0:
            raise ValueError("The items of split_size must sum to 1.0")

        self.sector_id = sector_id
        self.n_sectors = n_sectors
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.split_size = split_size
        self.drop_binary_features = drop_binary_features
        self.set_type = self.split_map[split]
        self.scale = scale
        self.is_pretrain = is_pretrain

        self.sector_map: dict[str, int] = {}
        self.scalers: list[StandardScaler] = []
        self.__load_data()

    @property
    def n_instances(self) -> int:
        return len(self.sector_map)

    @property
    def instance_size(self) -> int:
        return self.data_x.shape[1] - self.seq_len - self.pred_len + 1

    def __load_data(self):
        filepath = "datasets/telenor_sampled.pkl"
        df_raw: pd.DataFrame = np.load(filepath, allow_pickle=True)

        if self.sector_id is not None:
            df_raw = df_raw[df_raw["sector_id"] == self.sector_id]

        if self.n_sectors is not None:
            np.random.seed(42)
            random.seed(42)
            unique_sectors = df_raw["sector_id"].unique()
            sectors = np.random.choice(
                unique_sectors,
                min(
                    self.n_sectors,
                    unique_sectors.shape[0],
                ),
                replace=False,
            )
            df_raw = df_raw.loc[df_raw["sector_id"].isin(sectors)]
            print(df_raw["sector_id"].nunique())

        if self.drop_binary_features:
            df_raw = df_raw.drop(columns=self.binary_features)

        categorical_columns = df_raw.select_dtypes("object").columns.tolist()

        self.scalers = []
        all_data = []
        for i, item in enumerate(df_raw.groupby("sector_id")):
            sector_id, group = item
            self.sector_map[sector_id] = i
            group = group.drop(columns=categorical_columns)

            n_timesteps = group.shape[0]
            train_size, val_size, _ = self.split_size
            border1s = (
                0,
                int(n_timesteps * train_size) - self.seq_len,
                int(n_timesteps * (train_size + val_size)) - self.seq_len,
            )
            border2s = (
                int(n_timesteps * train_size),
                int(n_timesteps * (train_size + val_size)),
                n_timesteps,
            )
            border1 = border1s[self.set_type]
            border2 = border2s[self.set_type]

            if self.scale:
                scaler = StandardScaler().fit(group[border1s[0] : border2s[0]])
                data = scaler.transform(group)
            else:
                data = group.values

            all_data.append(data[border1:border2])

        self.data_x = np.array(all_data)
        self.data_y = np.array(all_data)

    def __getitem__(self, index) -> Tensor:
        instance = index // self.instance_size
        index = index % self.instance_size
        if self.is_pretrain:
            overlap = self.seq_len // 2
            l_start = max(0, index - overlap)
            l_end = l_start + self.seq_len
            r_start = l_end - overlap
            r_end = r_start + self.seq_len

            seq_x = self.data_x[instance, l_start:l_end]
            seq_y = self.data_y[instance, r_start:r_end]
        else:
            s_begin = index
            s_end = s_begin + self.seq_len
            r_begin = s_end
            r_end = r_begin + self.pred_len

            seq_x = self.data_x[instance, s_begin:s_end]
            seq_y = self.data_y[instance, r_begin:r_end]
        return _numpy(seq_x), _numpy(seq_y)

    def __len__(self) -> int:
        return self.n_instances * self.instance_size
