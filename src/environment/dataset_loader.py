"""
dataset_loader.py
Dataset handling, anomaly filtering, temporal splitting, and DataLoader construction
for genuine Super Mario World RAM telemetry for statistical and PINN models.
"""

from typing import Dict, Optional, Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from src.utils.seed import set_global_seed


class SMWTransitionDataset(Dataset):
    """
    PyTorch Dataset for single-step transitions (s_t, a_t, s_{t+1}).
    """

    def __init__(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        next_states: np.ndarray,
    ):
        self.states = torch.tensor(states, dtype=torch.float32)
        self.actions = torch.tensor(actions, dtype=torch.float32)
        self.next_states = torch.tensor(next_states, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.states[idx], self.actions[idx], self.next_states[idx]


class SMWSequenceDataset(Dataset):
    """
    PyTorch Dataset for sequential models (LSTM) processing sliding windows of length K.
    """

    def __init__(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        next_states: np.ndarray,
        episodes: np.ndarray,
        seq_len: int = 10,
    ):
        self.seq_len = seq_len
        self.state_seqs = []
        self.action_seqs = []
        self.target_states = []

        unique_eps = np.unique(episodes)
        for ep in unique_eps:
            mask = episodes == ep
            ep_s = states[mask]
            ep_a = actions[mask]
            ep_ns = next_states[mask]

            n_samples = len(ep_s)
            if n_samples <= seq_len:
                continue

            for i in range(n_samples - seq_len):
                self.state_seqs.append(ep_s[i : i + seq_len])
                self.action_seqs.append(ep_a[i : i + seq_len])
                self.target_states.append(ep_ns[i + seq_len - 1])

        self.state_seqs = torch.tensor(np.array(self.state_seqs), dtype=torch.float32)
        self.action_seqs = torch.tensor(np.array(self.action_seqs), dtype=torch.float32)
        self.target_states = torch.tensor(np.array(self.target_states), dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.state_seqs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.state_seqs[idx], self.action_seqs[idx], self.target_states[idx]


def load_and_preprocess_data(
    dataset_path: str = "data/raw/smw_gameplay_dataset.npz",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """
    Loads raw WRAM telemetry dataset, filters anomalous pit-death frames,
    and performs episodic temporal partitioning into Train, Val, and Test splits.
    """
    raw_data = np.load(dataset_path)
    states = raw_data["states"]
    actions = raw_data["actions"]
    next_states = raw_data["next_states"]
    episodes = raw_data["episodes"]

    # 1. Filter out-of-bounds death anomalies (pit falls where Y > 500 or Y < 0)
    valid_mask = (
        (states[:, 1] >= 0)
        & (states[:, 1] <= 500)
        & (next_states[:, 1] >= 0)
        & (next_states[:, 1] <= 500)
    )

    states = states[valid_mask]
    actions = actions[valid_mask]
    next_states = next_states[valid_mask]
    episodes = episodes[valid_mask]

    # 2. Episodic splitting to preserve contiguous temporal trajectory dynamics
    unique_eps = np.unique(episodes)
    set_global_seed(seed)
    shuffled_eps = np.random.permutation(unique_eps)

    n_eps = len(shuffled_eps)
    n_train = max(1, int(n_eps * train_ratio))
    n_val = max(1, int(n_eps * val_ratio))

    train_eps = shuffled_eps[:n_train]
    val_eps = shuffled_eps[n_train : n_train + n_val]
    test_eps = shuffled_eps[n_train + n_val :]
    if len(test_eps) == 0:
        test_eps = val_eps  # fallback if episode count is low

    train_mask = np.isin(episodes, train_eps)
    val_mask = np.isin(episodes, val_eps)
    test_mask = np.isin(episodes, test_eps)

    return {
        "train_states": states[train_mask],
        "train_actions": actions[train_mask],
        "train_next_states": next_states[train_mask],
        "train_episodes": episodes[train_mask],
        "val_states": states[val_mask],
        "val_actions": actions[val_mask],
        "val_next_states": next_states[val_mask],
        "val_episodes": episodes[val_mask],
        "test_states": states[test_mask],
        "test_actions": actions[test_mask],
        "test_next_states": next_states[test_mask],
        "test_episodes": episodes[test_mask],
    }


def create_dataloaders(
    data_dict: Dict[str, np.ndarray],
    batch_size: int = 128,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Constructs standard PyTorch DataLoaders for single-step transitions."""
    train_ds = SMWTransitionDataset(
        data_dict["train_states"],
        data_dict["train_actions"],
        data_dict["train_next_states"],
    )
    val_ds = SMWTransitionDataset(
        data_dict["val_states"],
        data_dict["val_actions"],
        data_dict["val_next_states"],
    )
    test_ds = SMWTransitionDataset(
        data_dict["test_states"],
        data_dict["test_actions"],
        data_dict["test_next_states"],
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader
