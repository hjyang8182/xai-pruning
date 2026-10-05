from typing import Optional, Literal, Union, Callable
from os import path, listdir, makedirs

import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from tqdm import tqdm, trange
import math
import tempfile
import shutil


# Dataset that is loads two datasets in parallel. 
class PDataset(Dataset):
    def __init__(self, *its: list[Union[Dataset, torch.Tensor, list]], list_to_tensor: bool = True):
        assert len(its) > 0, "At least one sequence must be given"
        assert all(self.get_length(its[0]) == self.get_length(it) for it in its), \
            "Provided iterables needs to be from same size. "
        self.its = its
        self.list_to_tensor = list_to_tensor
    
    def get_length(self, it):
        if isinstance(it, Dataset):
            return len(it)
        elif torch.is_tensor(it):
            return it.shape[0]
        else:
            return len(it)
    
    def __len__(self):
        return self.get_length(self.its[0])
    
    def __getitem__(self, idx):
        res = []
        for i in range(len(self.its)):
            it = self.its[i][idx]
            if isinstance(it, list) and self.list_to_tensor:
                it = torch.tensor(it)
            res.append(it)
        return res

# Class implementing a dataset that loads cached activations either from
# a single memmapped file (fast path, written by raw_concept_sims) or,
# for caches computed before that existed, from one .pth file per sample
# in a given directory in alphabetic order.
class MemTensorDataset(Dataset):
    def __init__(self, root: str, normalize=False, mean: Optional[torch.Tensor] = None, std: Optional[torch.Tensor] = None):
        self.root = root
        self.normalize = normalize
        meta_path = path.join(root, "meta.pt")
        self.legacy = not path.exists(meta_path)

        if not self.legacy:
            meta = torch.load(meta_path)
            self.n, self.d = meta["n"], meta["d"]
            self.data = np.memmap(path.join(root, "activations.dat"), dtype=np.float32, mode="r", shape=(self.n, self.d))
        else:
            if root[-1] == "/":
                root = root[:-1]
            last_dir = root.split("/")[-1]
            self.samples = [f for f in listdir(root) if f.endswith(".pth") and "activations" in f]
            def key(x: str) -> int:
                x = x.removeprefix(last_dir + "_")
                x = x.removesuffix(".pth")
                return int(x)
            self.samples.sort(key=key)

        if self.normalize:
            if mean is None or std is None:
                mean_path, std_path = path.join(root, "mean.pt"), path.join(root, "std.pt")
                if not path.exists(mean_path) or not path.exists(std_path):
                    if not self.legacy:
                        # Single sequential read over the memmap instead of one
                        # torch.load() per sample.
                        arr = np.asarray(self.data)
                        mean = torch.from_numpy(arr.mean(axis=0))
                        std = torch.from_numpy(arr.std(axis=0))
                        std = torch.where(std == 0, torch.ones_like(std), std)
                    else:
                        batch_size = 64
                        size = torch.load(path.join(self.root, self.samples[0])).size()
                        mean = torch.zeros(size)
                        meansq = torch.zeros(size)
                        for i in trange(math.ceil(len(self) / batch_size), leave=False):
                            start = i * batch_size
                            end = (i+1) * batch_size
                            if end > len(self):
                                end = len(self)
                            batch = torch.stack(
                                [torch.load(path.join(self.root, sam))
                                for sam in self.samples[start:end]])
                            mean += torch.sum(batch, dim=0)
                            meansq += torch.sum(batch**2, dim=0)
                        mean = mean / len(self)
                        meansq = meansq / len(self)
                        std = torch.sqrt(meansq - mean**2)
                        if torch.any(torch.isnan(std)):
                            diff = meansq - mean**2
                            std = torch.sqrt(torch.clamp(diff, min=diff[diff>0].min()))
                    assert not torch.any(torch.isnan(mean))
                    torch.save(mean, mean_path)
                    assert not torch.any(torch.isnan(std))
                    torch.save(std, std_path)
                else:
                    mean = torch.load(mean_path)
                    std = torch.load(std_path)
            self.mean = mean
            self.std = std

    def __len__(self):
        return len(self.samples) if self.legacy else self.n

    def __getitem__(self, idx):
        if self.legacy:
            tens = torch.load(path.join(self.root, self.samples[idx]))
        else:
            tens = torch.from_numpy(np.array(self.data[idx]))
        if self.normalize:
            tens = (tens - self.mean)  / self.std
        return tens


@torch.no_grad()
def raw_concept_sims(h: Union[np.ndarray, torch.Tensor], 
                     dataset: Dataset, 
                     backbone: Union[nn.Module, Callable], 
                     batch_size: int, 
                     device: Literal['cuda', 'cpu'], 
                     saved_activation_path: Optional[str] = None, 
                     data_label: Optional[str] = None,
                     normalize=False, 
                     mean=None, 
                     std=None) \
                        -> Dataset[torch.Tensor]:
    '''
    Compute raw concept similarities using cosine similarity. 

    Parameters
    ----------
    h: np.ndarray
        Concept vectors in shape (#concepts, p). 
    dataset: Dataset
        The dataset to compute the concept similarities on. 
    backbone: nn.Module | Callable
        The model backbone. 
    batch_size: int
        The batch_size in which the concept similarities are computed. 
    device: Literal['cuda', 'cpu']
        The device to compute the similarities on. 
    saved_activation_path: Optional[str] = None
        The folder where the activations of the current dataset and
        backbone can be/are saved. 
    data_label: Optional[str] = None
        The data_label (train or test) for the dataset to identify the
        correct pre_computed concept similarities. 
    normalize: bool = False
        Wheter to normalize the concept similarities concept-wise. If mean and
        std is given, these values are used, otherwise mean and std will be
        computed on the given dataset. 
    mean: Optional[torch.Tensor] = None
    std: Optional[torch.Tensor] = None

    Returns
    -------
    sim: Dataset[torch.Tensor]
        Dataset containing the concept similarities of each image in 
        given dataset (in the same order). 
    '''

    assert ((saved_activation_path is None and data_label is None) or
            (saved_activation_path is not None and data_label is not None)), \
            "saved_activation_path and data_label must be both None or not None. "
    
    save = data_label is not None
    if not save:
        saved_activation_path = tempfile.gettempdir()
        data_label = "tmp"

    save_name = f"saved_{data_label}_activations"
    saving_dir = path.join(saved_activation_path, save_name)
    meta_path = path.join(saving_dir, "meta.pt")
    data_path = path.join(saving_dir, "activations.dat")
    progress_path = path.join(saving_dir, "progress.pt")
    n_total = len(dataset)

    # Check if a cache already exists and is complete. If so, read from it
    # directly (either the fast single-memmap format, or, for caches computed
    # before that existed, the legacy one-file-per-sample format).
    start_row = 0
    if save and path.exists(saving_dir):
        legacy_files = [f for f in listdir(saving_dir) if "activations" in f and f.endswith(".pth")]
        if legacy_files and len(legacy_files) == n_total:
            return MemTensorDataset(saving_dir, normalize=normalize, mean=mean, std=std)
        if path.exists(meta_path):
            meta = torch.load(meta_path)
            if meta["n"] == n_total:
                start_row = torch.load(progress_path) if path.exists(progress_path) else 0
                if start_row >= n_total:
                    return MemTensorDataset(saving_dir, normalize=normalize, mean=mean, std=std)
            else:
                # Stale cache (different dataset size) -- start over.
                shutil.rmtree(saving_dir)
                makedirs(saving_dir)
        else:
            # Partial leftovers from an interrupted run -- start over rather
            # than trying to stitch mismatched formats together.
            shutil.rmtree(saving_dir)
            makedirs(saving_dir)
    elif save:
        makedirs(saving_dir, exist_ok=True)
    else:
        if path.exists(saving_dir):
            shutil.rmtree(saving_dir)
        makedirs(saving_dir)

    # Calculate the concept similarities.

    # Convert CAVs to normalized tensor.
    h = h if torch.is_tensor(h) else torch.tensor(h)
    h = h.to(device)
    h /= torch.norm(h, dim=1, keepdim=True)
    n_concepts = h.shape[0]

    if not path.exists(meta_path):
        torch.save({"n": n_total, "d": n_concepts}, meta_path)
    mmap = np.memmap(data_path, dtype=np.float32,
                      mode="r+" if path.exists(data_path) else "w+",
                      shape=(n_total, n_concepts))

    # Load the data from dataset.
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                              num_workers=8, pin_memory=(device != "cpu"))

    if isinstance(backbone, nn.Module):
        backbone.to(device)
        backbone.eval()

    # Compute concept similarities.
    row = 0
    for X_batch, _ in tqdm(data_loader, leave=False):
        width = X_batch.shape[0]
        if row + width <= start_row:
            row += width
            continue

        out = backbone(X_batch.to(device))

        if len(out.shape) == 4:
            out = torch.mean(out, dim=(2, 3))

        out = out / torch.norm(out, dim=1, keepdim=True)
        out = out.type(h.dtype)

        out = torch.matmul(out, h.T)

        mmap[row:row+width] = out.cpu().float().numpy()
        row += width
        if save:
            torch.save(row, progress_path)
    mmap.flush()

    return MemTensorDataset(saving_dir, normalize=normalize, mean=mean, std=std)