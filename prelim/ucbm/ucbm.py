from typing import Literal, Union, Optional, Callable
import numpy as np
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm, trange
from scipy.special import softmax
from sklearn.metrics import roc_auc_score
from copy import deepcopy

from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
from torchvision.datasets import ImageFolder
from concept_auxiliaries.concept_auxiliaries import raw_concept_sims, PDataset

import sys as _sys
_sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from concept_usage import concept_usage_report  # noqa: E402  (shared backbone-agnostic metric at repo root)
from gating import prune_and_refit as _prune_and_refit, masked_accuracy as _masked_accuracy, evaluate_head as _evaluate_head, hard_gate_ste  # noqa: E402
from torcheval.metrics.functional import multilabel_accuracy, \
    multiclass_accuracy, multiclass_auprc, multilabel_auprc, binary_auroc, \
    multiclass_auroc, binary_auprc


################################################################################
#                                                                              #
#                                  LeakyReLU                                   #
#         ReLU-similar function, which is not killing the gradient.            #
#                                                                              #
################################################################################
class LeakyReLU(torch.autograd.Function):
    @staticmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.neg = x < 0
        return x.clamp(min=0.0)
    
    @staticmethod
    def backward(self, grad_output: torch.Tensor):
        grad_input = grad_output.clone()
        grad_input[self.neg] *= 0.1
        return grad_input



################################################################################
#                                                                              #
#                                   JumpReLU                                   #
#                 JumpReLU with straight-through estimator.                    #
#                Following: https://arxiv.org/abs/2407.14435                   #
#                                                                              #
################################################################################
class RectangleFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return ((x > -0.5) & (x < 0.5)).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[(x <= -0.5) | (x >= 0.5)] = 0
        return grad_input

def rectangle(x: torch.Tensor) -> torch.Tensor:
    return RectangleFunction.apply(x)

class _JumpReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, 
                x: torch.Tensor, 
                threshold: torch.Tensor, 
                bandwidth: float) -> torch.Tensor:
        ctx.save_for_backward(x, threshold)
        ctx.bandwidth = bandwidth
        return x * (x > threshold).to(x.dtype)

    @staticmethod
    def backward(ctx, output_grad: torch.Tensor): # ste
        x, threshold = ctx.saved_tensors
        bandwidth = ctx.bandwidth
        x_grad = (x > threshold).to(x.dtype) * output_grad
        rect_value = rectangle((x - threshold) / bandwidth)
        threshold_grad = -(threshold / bandwidth) * rect_value * output_grad
        return x_grad, threshold_grad, None

class JumpReLU(nn.Module):
    def __init__(self, 
                 num_concepts: int, 
                 threshold_init: Optional[torch.Tensor] = None, 
                 bandwidth: float = 1e-3):
        super(JumpReLU, self).__init__()
        self.log_threshold = nn.Parameter(-10*torch.ones(num_concepts, 
                                                         requires_grad=True))
        if threshold_init is not None:
            assert threshold_init.numel() == num_concepts, \
                f"Init threshold is of dimension {threshold_init.size()}" + \
                f", but should be of dimension {num_concepts}. "
            self.log_threshold = threshold_init
        self.bandwidth = bandwidth

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _JumpReLU.apply(
            x, 
            self.log_threshold.exp(),  # exp ensures positive threshold
            self.bandwidth
            )



################################################################################
#                                                                              #
#                                    L0-Loss                                   #
#           L0-similar function which is not killing the gradient.             #
#                                                                              #
################################################################################
class _StepFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, 
                x: torch.Tensor, 
                threshold: torch.Tensor, 
                bandwidth: float) -> torch.Tensor:
        ctx.save_for_backward(x, threshold)
        ctx.bandwidth = bandwidth
        return (x > threshold).to(x.dtype)

    @staticmethod
    def backward(ctx, output_grad: torch.Tensor): # ste
        x, threshold = ctx.saved_tensors
        bandwidth = ctx.bandwidth
        x_grad = torch.zeros_like(x)
        rect_value = rectangle((x - threshold) / bandwidth)
        threshold_grad = -(1.0 / bandwidth) * rect_value * output_grad
        return x_grad, threshold_grad, None

class StepFunction(nn.Module):
    def __init__(self):
        super(StepFunction, self).__init__()

    def forward(self, 
                x: torch.Tensor, 
                threshold: torch.Tensor, 
                bandwidth: float) -> torch.Tensor:
        return _StepFunction.apply(x, threshold, bandwidth)

step_function = StepFunction()
def l0_loss(x: torch.Tensor, 
            threshold: torch.Tensor, 
            bandwidth: float) -> torch.Tensor:
    out = step_function(x, threshold, bandwidth)
    return torch.sum(out, dim=-1).sum()


def l0_approx(x: torch.Tensor, 
              threshold: torch.Tensor, 
              a: int = 20) -> float:
    out = 1 / (1 + torch.exp(-a*(x - threshold)))
    return out.sum().item()



################################################################################
#                                                                              #
#                                 Elastic-Loss                                 #
#                                                                              #
################################################################################
def elastic_loss(weight_or_act: torch.Tensor, 
                 alpha: float = 0.99) -> torch.Tensor:
    l1 = weight_or_act.norm(p=1)
    l2 = (weight_or_act**2).sum()
    return 0.5 * (1 - alpha) * l2 + alpha * l1

def elastic_loss_weights(weight: torch.Tensor, 
                         alpha: float = 0.99) -> torch.Tensor:
    l1 = weight.norm(p=1)
    l2 = (weight**2).sum()
    return 0.5 * (1 - alpha) * l2 + alpha * l1

def elastic_loss_activations(act: torch.Tensor,
                             alpha: float = 0.99) -> torch.Tensor:
    l1 = act.norm(p=1, dim=-1)
    l2 = (act**2).sum(dim=-1)
    return torch.sum(0.5 * (1 - alpha) * l2 + alpha * l1)


def gate_temperature_schedule(epoch: int, epochs: int, start: float,
                              end: Optional[float] = None) -> float:
    '''
    Geometric (log-linear) decay of the gate's sigmoid temperature from
    `start` to `end` across `epochs` epochs, evaluated at `epoch`
    (0-indexed). Geometric rather than linear so the temperature spends
    proportionally similar time at each order of magnitude, e.g. as long
    going from 1.0 -> 0.3 as from 0.3 -> 0.09 -- avoids the fast part of the
    decay being crammed into the first few epochs the way a linear schedule
    would if start and end differ by an order of magnitude or more.

    Returns `start` unchanged if `end` is None or `epochs <= 1`, i.e. a
    fixed (non-annealed) temperature throughout -- this is the previous
    fixed-gate_temperature behavior, so leaving `end=None` reproduces it
    exactly.
    '''
    if end is None or epochs <= 1:
        return start
    frac = epoch / (epochs - 1)
    return start * (end / start) ** frac


################################################################################
#                                                                              #
#                                 TopK-Module                                  #
#                  Module that keeps only k largest values.                    #
#      Implementation from  https://github.com/openai/sparse_autoencoder       #
################################################################################
class TopK(nn.Module):
    def __init__(self, 
                 k: int, 
                 postact_fn: Callable = nn.ReLU()):
        super().__init__()
        self.k = k
        self.postact_fn = postact_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        topk = torch.topk(x, k=self.k, dim=-1)
        values = self.postact_fn(topk.values)
        # make all other values 0
        result = torch.zeros_like(x)
        result.scatter_(-1, topk.indices, values)
        return result

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        state_dict = super().state_dict(destination, prefix, keep_vars)
        state_dict.update(
            {prefix + "k": self.k, 
             prefix + "postact_fn": self.postact_fn.__class__.__name__})
        return state_dict

    @classmethod
    def from_state_dict(cls, 
                        state_dict: dict[str, torch.Tensor], 
                        strict: bool = True) -> "TopK":
        k = state_dict["k"]
        postact_fn = ACTIVATIONS_CLASSES[state_dict["postact_fn"]]()
        return cls(k=k, postact_fn=postact_fn)

ACTIVATIONS_CLASSES = {
    "ReLU": nn.ReLU,
    "Identity": nn.Identity,
    "TopK": TopK,
}




################################################################################
#                                                                              #
#                                  Classifier                                  #
#             Interpretable classifier from concepts to classes.               #
#                                                                              #
################################################################################
class Classifier(nn.Module):
    def __init__(self, 
                 num_concepts: int, 
                 num_classes: int, 
                 relu: Literal["no", "ReLU", "jumpReLU"] = "ReLU", 
                 scale: Literal["learn", "no"] = "no", 
                 bias: Literal["learn", "no"] = "no", 
                 dropout_p: float = 0,
                 k: int = -1,
                 jumpReLU_threshold_init: Optional[torch.Tensor] = None,
                 gated: bool = False,
                 gate_temperature: float = 1.0,
                 gate_forward: Literal["soft", "hard"] = "soft"):
        '''
        Interpretable (linear) classifier from concept space to
        classification space.

        num_concepts: int
            number of concepts in concept space
        num_classes: int
            number of classes in classification space
        relu: Literal["no", "ReLU", "jumpReLU"] = "ReLU"
            relu function to use after gating
        scale: Literal["learn", "no"] = "no"
            scale to scale the cosine similarities,
            set "learn" to learn scale and "no" to disable
        bias: Literal["learn", "no"] = "no"
            bias to substract from scaled cosine similarities
            set "learn" to learn bias and "no" to disable
        dropout_p: float = 0
            dropout rate from concept dropout
        k: int = -1
            If k >= 0 TopK is used instead of gating, else nothing happens
        jumpReLU_threshold_init: Optional[torch.Tensor] = None
            Init value of jumpReLU
        gated: bool = False
            If True, a learned per-concept sigmoid gate is applied on top
            of the relu/jumpReLU/TopK selection above, before the linear
            layer.
        gate_temperature: float = 1.0
            Divides gate_logits before the sigmoid: sigmoid(gate_logits / T).
            T < 1 sharpens the sigmoid, pushing gate values toward 0/1 (more
            bimodal) for the same spread of logits; T > 1 softens it. T = 1
            is the plain sigmoid, unchanged from before. Only affects
            anything when gated=True.
        '''
        super().__init__()
        self.relu = relu
        if relu == "jumpReLU":
            self._jumpReLU = JumpReLU(num_concepts, jumpReLU_threshold_init)
        self.scale_method = scale
        self.bias_method = bias
        self.dropout = (dropout_p > 0)
        self.dropout_layer = nn.Dropout(p=dropout_p)
        self.k = k
        if self.k >= 0:
            self.top_k = TopK(self.k)
        if self.scale_method == "learn":
            self.log_scaling = nn.Parameter(
                torch.zeros(num_concepts, requires_grad=True))
        if self.bias_method == "learn":
            self.log_offset = nn.Parameter(
                -10*torch.ones(num_concepts, requires_grad=True))
        self.gated = gated
        self.gate_temperature = gate_temperature
        # "soft": multiply by sigmoid(gate); "hard": multiply by 1[sigmoid > 0.5] with a
        # straight-through gradient, so the trained model is exactly the masked one and the
        # head cannot absorb a shrinking gate (see gating.GatedProbe).
        self.gate_forward = gate_forward
        if self.gated:
            self.gate_logits = nn.Parameter(torch.ones(num_concepts) * 2.0)
        self.linear = nn.Linear(num_concepts, num_classes)

    def gate_probs(self) -> torch.Tensor:
        '''
        sigmoid(gate_logits / gate_temperature), the actual per-concept gate
        value used everywhere (forward pass, the gate's sparsity penalty,
        and open/closed thresholding at 0.5 -- unaffected by temperature
        since sigmoid(0) = 0.5 regardless of T). Only valid if gated=True.
        '''
        assert self.gated, "Classifier was not constructed with gated=True."
        return torch.sigmoid(self.gate_logits / self.gate_temperature)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # input-dependent concept selection
        if self.scale_method == "learn":
            x = self.log_scaling.exp() * x  # exp ensures positive scale

        if self.bias_method == "learn":
            x = x - self.log_offset.exp()  # exp ensures positive bias

        if self.relu == "ReLU":
            gated = F.relu(x)
        elif self.relu == "jumpReLU":
            gated = self._jumpReLU(x)
        elif self.k >= 0:
            gated = self.top_k(x)
        elif self.relu == "no":
            gated = x
        else:
            raise NotImplementedError

        # learned per-concept gate, layered on top of the selection above
        if self.gated:
            g = self.gate_probs()
            gated = gated * (hard_gate_ste(g) if self.gate_forward == "hard" else g)

        # concept dropout
        if self.dropout:
            mask = torch.ones_like(gated)
            mask = self.dropout_layer(mask)
            gated = gated * mask
            x = x * mask

        # sparse linear layer
        out = self.linear(gated)
        return out, gated, x


################################################################################
#                                                                              #
#                                     UCBM                                     #
#                    unsupervised concept bottleneck model                     #
#                                                                              #
################################################################################
class UCBM:
    '''
    Class implementing an unsupervised concept bottleneck model. 

    Parameters
    ----------
    backbone
        model backbone to compute the embeddings
    h: torch.Tensor | np.ndarray
        matrix of shape (n, p) containing all n concept activation vectors
    batch_size: int
        batch size to use
    epochs: int
        amount of training epochs
    lam_pi: float
        regularization strength of the elastic-net penalty on the concept
        selector's output pi(x) (ReLU/jumpReLU/TopK), independent of gating
    lambda_gate: float
        regularization strength of the learned per-concept gate's own
        sparsity penalty (sum of sigmoid(gate_logits)), applied on top of
        the Classifier's other regularizers (lam_pi, lam_w) when gated=True
    lam_w: float
        regularization strength of penalization on linear weights
    dropout_p: float
        concept dropout rate
    learning_rate: float
        learning rate
    relu: Literal["no", "ReLU", "jumpReLU"]
        relu to use for gating
    scale_mode: Literal['learn', 'no']
        scale for concept similarities
    bias_mode: Literal['learn', 'no']
        bias for concept similarities
    normalize: bool
        normalize the cosine similarities of each concept? 
    k: int
        k to use for TopK module, if -1 no TopK module is used
    device: Literal['cuda', 'cpu']
        device to use for computations
    gate_temperature: float
        temperature dividing gate_logits before the sigmoid (only used when
        gated=True); T < 1 sharpens the gate toward 0/1 (more bimodal), T = 1
        is the plain sigmoid. See Classifier.gate_probs(). If
        gate_temperature_final is given, this is the *starting* temperature
        of a geometric anneal rather than a fixed value.
    gate_temperature_final: Optional[float]
        If given (and gated=True), the gate temperature is annealed
        geometrically from gate_temperature down (or up) to this value over
        the course of fit()'s epochs -- see gate_temperature_schedule().
        Leave as None to keep gate_temperature fixed throughout, as before.
    '''

    def __init__(self,
                 backbone,
                 h: Union[torch.Tensor, np.ndarray],
                 batch_size: int,
                 epochs: int,
                 lam_pi: float,
                 lambda_gate: float,
                 lam_w: float,
                 dropout_p: float,
                 learning_rate: float,
                 relu: Literal["no", "ReLU", "jumpReLU"],
                 scale_mode: Literal['learn', 'no'],
                 bias_mode: Literal['learn', 'no'],
                 normalize: bool,
                 k: int,
                 device: Literal['cuda', 'cpu'],
                 gated: bool,
                 gate_temperature: float = 1.0,
                 gate_temperature_final: Optional[float] = None,
                 gate_forward: Literal["soft", "hard"] = "soft"):

        self._backbone = backbone
        if not torch.is_tensor(h):
            h = torch.tensor(h)
        self._num_concepts = h.shape[0]
        self._h = h.to(device)
        self._h = self._h / torch.norm(self._h, dim=1, keepdim=True)
        self._batch_size = batch_size
        self._lr = learning_rate
        self._device = device

        self._epochs = epochs
        self._lam_pi = lam_pi
        self._lambda_gate = lambda_gate
        self._lam_w = lam_w
        self._dropout_p = dropout_p
        self._relu = relu
        self._scale_mode = scale_mode
        self._bias_mode = bias_mode
        self._normalize = normalize
        self._k = k
        self._gated = gated
        self._gate_temperature = gate_temperature
        self._gate_temperature_final = gate_temperature_final
        self._gate_forward = gate_forward

    @torch.no_grad()
    def _get_concept_embeddings(self, 
                                dataset: Dataset, 
                                saved_activation_path: Optional[str] = None, 
                                data_label: Optional[str] = None, 
                                normalize=False, 
                                mean=None, 
                                std=None) \
                                    -> Dataset[torch.Tensor]:
        '''
        Compute and save the concept embeddings using the cosine similarity or 
        load them from the given information. 

        Parameters
        ----------
        dataset: Dataset
            dataset to compute concept embeddings on
        saved_activation_path: Optional[str] = None
            folder where the activations of the current dataset and
            backbone can be/are saved
        data_label: Optional[str] = None
            data_label (train or test) for the dataset to identify the
            correct pre_computed concept similarities
        
        Returns
        -------
        sims: Dataset[torch.Tensor]
            Dataset containing the concept similarities of each image in 
            given dataset (in the same order) 
        '''

        return raw_concept_sims(self._h, 
                                dataset, 
                                self._backbone, 
                                self._batch_size, 
                                self._device, 
                                saved_activation_path, 
                                data_label, 
                                normalize=normalize, 
                                mean=mean, 
                                std=std)
    
    def fit(self, 
            training_set: ImageFolder, 
            saved_activation_path: str, 
            test_set: Optional[ImageFolder] = None, 
            verbose: bool = True, 
            cocostuff_training: bool = False):
        '''
        Function fit UCBM to given dataset. 

        Parameters
        ----------
        training_set: ImageFolder
            data_set to train UCBM on
        saved_activation_path: str
            path to save concept activations
        test_set: Optional[ImageFolder] = None
            If given, test accuracy is printed to stdout each epoch
        verbose: bool = True
            print progress to to stdout? 
        '''

        # Load the concept activations. 
        embeddings = self._get_concept_embeddings(
            training_set, 
            saved_activation_path, 
            "train", 
            normalize=self._normalize, 
            mean=None, 
            std=None)
        self._mean = embeddings.mean if self._normalize else None
        self._std = embeddings.std if self._normalize else None

        if verbose:
            print("Loaded concept activations of training dataset...")

        # Function that returns the indexes of a sequence, that are label
        # with the given class_id. 
        def indices_tensor(targets, class_id):
            return (torch.Tensor(targets) == class_id).nonzero().reshape(-1)

        # Load scale and bias values. 
        self._num_classes = len(training_set.classes)
        self._multilabel = isinstance(training_set.targets[0], list)
        
        # Load the model.
        self._classifier = Classifier(
            self._num_concepts,
            self._num_classes,
            self._relu,
            self._scale_mode,
            self._bias_mode,
            self._dropout_p,
            self._k,
            gated=self._gated,
            gate_temperature=self._gate_temperature,
            gate_forward=self._gate_forward)
        self._classifier = self._classifier.to(self._device)

        # Load stuff required for training the model. 
        loss_fn = nn.BCEWithLogitsLoss() if self._multilabel else nn.CrossEntropyLoss()
        optimizer = optim.Adam(self._classifier.parameters(), lr=self._lr)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer, T_max=self._epochs)
        
        # Load data
        dset = PDataset(embeddings, training_set.targets)
        if not cocostuff_training:
            data_loader = DataLoader(dset, self._batch_size, shuffle=True, 
                                    num_workers=8)
            
            # Train the model
            for i in trange(self._epochs, leave=False):
                self._classifier.train()
                if self._gated:
                    self._classifier.gate_temperature = gate_temperature_schedule(
                        i, self._epochs, self._gate_temperature, self._gate_temperature_final)
                corr, n_samples = 0, 0
                if self._relu == "jumpReLU":
                    print(self._classifier._jumpReLU.log_threshold.exp().mean())
                    print(self._classifier._jumpReLU.log_threshold.mean())
                for X_batch, y_batch in tqdm(data_loader, leave=False):
                    y_batch = y_batch.to(self._device)
                    out = X_batch.to(self._device)
                    
                    # Train the model.
                    optimizer.zero_grad()
                    y_pred, after_gate, before_gate = self._classifier(out)

                    # task-specific loss
                    if self._multilabel:
                        loss = loss_fn(y_pred, y_batch.to(y_pred.dtype))
                    else:
                        loss = loss_fn(y_pred, y_batch)

                    # existing selector's activation sparsity penalty --
                    # applies regardless of gating, since the selector always
                    # runs now (the learned gate is layered on top of it)
                    if self._lam_pi != 0:
                        if self._relu != "jumpReLU":
                            loss += self._lam_pi * elastic_loss_activations(after_gate)
                        else:
                            loss += self._lam_pi * l0_loss(before_gate, self._classifier._jumpReLU.log_threshold.exp(), self._classifier._jumpReLU.bandwidth)

                    # weight sparsity penalty -- unchanged regardless of gating;
                    # the gate's own sparsity penalty is added on top of it, not
                    # swapped in for it, so lam_w means the same thing whether or
                    # not gated=True. If the gate needs a softer weight penalty
                    # to train well, that's what --lam_w is for.
                    if self._lam_w != 0:
                        loss += self._lam_w * elastic_loss_weights(self._classifier.linear.weight)
                    if self._gated and self._lambda_gate != 0:
                        loss += self._lambda_gate * self._classifier.gate_probs().sum()

                    loss.backward()
                    optimizer.step()

                    # Compute training accuracy
                    if self._multilabel:
                        corr += y_pred.shape[0] * multilabel_accuracy(
                            torch.sigmoid(y_pred), y_batch, criteria="hamming")
                    else:
                        corr += y_pred.shape[0] * multiclass_accuracy(
                            torch.argmax(y_pred, dim=1), y_batch)
                    n_samples += y_pred.shape[0]
                
                # src/train.py::train_gated_probe trains gated probes at a
                # constant LR (no scheduler) -- decaying LR here would let
                # early, CE-dominated epochs push gates open and then freeze
                # them there, since by the time LR is low enough for the
                # (weak) sparsity term to matter more there's no step size
                # left to close them back down.
                if not self._gated:
                    lr_scheduler.step()
                if test_set:
                    self._classifier.eval()
                    test_acc = self.get_evaluation_metric(
                        test_set, saved_activation_path=saved_activation_path, data_label="test", metric=["acc"])["acc"]
                if verbose:
                    print(f"Epoch {i+1} / {self._epochs} - " +
                        f"train acc: {100 * corr / n_samples:.2f}%" +
                        (f", test acc: {100 * test_acc:.2f}%" if test_set else ""))
        else:
            # Train the model
            for i in trange(self._epochs, leave=False):
                self._classifier.train()
                if self._gated:
                    self._classifier.gate_temperature = gate_temperature_schedule(
                        i, self._epochs, self._gate_temperature, self._gate_temperature_final)
                corr = 0

                target_classes = deepcopy(list(training_set.class_to_idx.values()))
                np.random.shuffle(target_classes)
                all_subsets = []
                for target_cls in target_classes:
                    optimizer.zero_grad()
                    tar = torch.tensor(training_set.targets)[:,target_cls]
                    pos_idcs = indices_tensor(tar, 1).detach().numpy()
                    neg_idcs = indices_tensor(tar, 0).detach().numpy()
                    n = self._batch_size // 2
                    pos_samples = np.random.choice(pos_idcs, n, replace=len(pos_idcs) <= n)
                    neg_samples = np.random.choice(neg_idcs, n, replace=len(neg_idcs) <= n)

                    subset_dset = Subset(dset, pos_samples.tolist() + neg_samples.tolist())
                    assert len(subset_dset) == self._batch_size
                    all_subsets.append(subset_dset)
                data_loader = DataLoader(torch.utils.data.ConcatDataset(all_subsets), batch_size=self._batch_size, shuffle=False, num_workers=8)

                for idx, (X_batch, y_batch) in tqdm(enumerate(data_loader), leave=False, total=len(target_classes)):
                    y_batch = y_batch.to(self._device)
                    out = X_batch.to(self._device)
                    y_pred, gate, _ = self._classifier(out)

                    # loss = loss_fn(y_pred, y_batch.to(y_pred.dtype))
                    loss = loss_fn(y_pred[:, target_classes[idx]], y_batch.to(y_pred.dtype)[:, target_classes[idx]]) # compute loss only on the target class output logit
                    if self._lam_pi != 0:
                        loss = loss + self._lam_pi * elastic_loss_activations(gate)

                    if self._lam_w != 0:
                        loss = loss + self._lam_w * elastic_loss_weights(self._classifier.linear.weight)
                    if self._gated and self._lambda_gate != 0:
                        loss = loss + self._lambda_gate * self._classifier.gate_probs().sum()

                    loss.backward()
                    optimizer.step()

                    corr += multilabel_accuracy(torch.sigmoid(y_pred), y_batch, criteria="hamming")
                
                if not self._gated:
                    lr_scheduler.step()
                if test_set:
                    self._classifier.eval()
                    test_acc = self.get_evaluation_metric(
                        test_set, saved_activation_path=saved_activation_path, data_label="test", metric=["acc"])["acc"]
                if verbose:
                    print(f"Epoch {i+1} / {self._epochs} - " +
                        f"train acc: {100 * corr / len(training_set.classes):.2f}%" +
                        (f", test acc: {100 * test_acc:.2f}%" if test_set else ""))


    
    @torch.no_grad()
    def predict(self, 
                imgs: torch.Tensor) \
                    -> tuple[torch.Tensor, torch.Tensor]:
        '''
        Function prediction the img classes and concept of imgs. 

        Paramters
        ---------
        imgs: torch.Tensor
            Input images in shape (n, #channel, width, height). 
        
        Returns
        -------
        y_pred: torch.Tensor
            Tensor containing the predicted classes. Shape (n). 
        concept values:
            Tensor of shape (n, #concepts) 
        '''

        assert hasattr(self, "_classifier"), "Model not yet fitted. "

        self._classifier.eval()

        out = self._backbone(imgs.to(self._device))

        if len(out.shape) == 4:
            out = torch.mean(out, dim=(2, 3))
            
        out = out / torch.norm(out, dim=1, keepdim=True)
        out = out.type(self._h.dtype)

        out = torch.matmul(out, self._h.T)

        if self._normalize:
            out = (out - self._mean.to(self._device)) / self._std.to(self._device)

        out, gate, _ = self._classifier(out)

        if self._multilabel:
            out = torch.sigmoid(out)

        out, gate = out.cpu(), gate.cpu()
        
        return out, gate
    
    @torch.no_grad()
    def get_evaluation_metric(self, 
                              dataset: ImageFolder, 
                              metric: list[Literal["acc", "auprc", "auroc", "auprc_pc"]] = ["acc"], 
                              saved_activation_path: Optional[str] = None, 
                              data_label: Optional[str] = None) \
                                -> dict[str, float]:
        '''
        Function that computes the accuracy of this model to the given dataset. 

        Parameters
        ----------
        dataset: ImageFolder
            Dataset for which the accuracy should be computed for. 
        metric: Literal["acc", "auprc", "auroc"]
        saved_activation_path: Optional[str] = None
            If not None, this is the path where the concept similarities 
            can be/are saved. 
        data_label: Optional[str] = None
            Label (train or test) from given dataset. 
        
        Returns
        -------
        accuracy: list[Optional[float]]
        '''

        if next(self._classifier.parameters()).device != self._device:  
            self._classifier.to(self._device)
        self._classifier.eval()

        # Load the concept activations. 
        embeddings = self._get_concept_embeddings(
            dataset, saved_activation_path, data_label, 
            self._normalize, self._mean, self._std)
        
        dset = PDataset(embeddings, dataset.targets)
        data_loader = DataLoader(dset, batch_size=self._batch_size, 
                                 shuffle=False, num_workers=8)
        y_pred = []
        y_true = []
        for X_batch, y_batch in data_loader:
            y_predb, _, _ = self._classifier(X_batch.to(self._device))
            if self._multilabel:
                y_predb = torch.sigmoid(y_predb)
            y_predb = y_predb.cpu()
            y_pred.append(y_predb)
            y_true.append(y_batch)
        
        y_pred = torch.cat(y_pred)
        y_true = torch.cat(y_true)

        def indices_tensor(targets, class_id):
            return (torch.Tensor(targets) == class_id).nonzero().reshape(-1)

        metrics = {}
        for me in metric:
            if me == "acc":
                if self._multilabel:
                    metrics[me] = multilabel_accuracy(y_pred, y_true, criteria="hamming").item()
                else:
                    metrics[me] = multiclass_accuracy(y_pred, y_true).item()
            elif me == "auprc" and self._multilabel:
                if self._multilabel:
                    # metrics[me] = multilabel_auprc(y_pred, y_true).item()
                    n = 500
                    auprc_pc = 0
                    for cls in dataset.class_to_idx.values():
                        tar = torch.tensor(dataset.targets)[:,cls]
                        pos_idcs = indices_tensor(tar, 1).detach().numpy()
                        neg_idcs = indices_tensor(tar, 0).detach().numpy()
                        pos_samples = np.random.choice(pos_idcs, n//2, len(pos_idcs) < n//2)
                        neg_samples = np.random.choice(neg_idcs, n//2, len(neg_idcs) < n//2)

                        samples = pos_samples.tolist() + neg_samples.tolist()

                        auprc_pc += binary_auprc(y_pred[samples, cls], y_true[samples, cls]).item()
                    auprc_pc /= len(dataset.class_to_idx)
                    metrics[me] = auprc_pc

                else:
                    metrics[me] = multiclass_auprc(y_pred, y_true).item()
            elif me == "auprc_pc" and self._multilabel:
                n = 500
                auprc_pc = []
                for cls in dataset.class_to_idx.values():
                    tar = torch.tensor(dataset.targets)[:,cls]
                    pos_idcs = indices_tensor(tar, 1).detach().numpy()
                    neg_idcs = indices_tensor(tar, 0).detach().numpy()
                    pos_samples = np.random.choice(pos_idcs, n//2, len(pos_idcs) < n//2)
                    neg_samples = np.random.choice(neg_idcs, n//2, len(neg_idcs) < n//2)

                    samples = pos_samples.tolist() + neg_samples.tolist()

                    auprc_pc.append(binary_auprc(y_pred[samples, cls], y_true[samples, cls]).item())
                
                metrics[me] = auprc_pc
            elif me == "auroc":
                if self._multilabel:
                    auroc = 0
                    n = y_pred.shape[1]
                    for i in range(n):
                        auroc += binary_auroc(y_pred[:,i], y_true[:,i]).item()
                    metrics[me] = auroc / n
                else:
                    if len(y_true.unique()) == 2:
                        # metrics[me] = binary_auroc(torch.argmax(y_pred, dim=1), y_true).item()
                        metrics[me] = roc_auc_score(y_true.numpy(), softmax(y_pred.numpy(), axis=1)[:, 1])
                    else:
                        metrics[me] = multiclass_auroc(y_pred, y_true, num_classes=len(dataset.classes)).item()
        return metrics
    
    @torch.no_grad()
    def compute_concept_similarities(self, 
                                     dataset: Dataset, 
                                     saved_activation_path: str, 
                                     data_label: str) -> torch.Tensor:
        '''
        Get concept simularities for given dataset. 

        Parameters
        ----------
        dataset: Dataset
            The dataset for which the concept similarities should be 
            computed for. 
        saved_activation_path: str
            Path where the concept similarities can be/are saved. 
        data_label: str
            Label (train or test) from given dataset. 

        Returns
        -------
        concept_similarities: torch.Tensor
            The concept similarities in shape (n, #concepts). 
        '''

        self._classifier.eval()

        # Load the concept activations. 
        embeddings = self._get_concept_embeddings(
            dataset, saved_activation_path, data_label, 
            self._normalize, self._mean, self._std)
        
        data_loader = DataLoader(embeddings, batch_size=self._batch_size, 
                                 shuffle=False, num_workers=8)
        sims = []
        for con in data_loader:
            _, sim, _ = self._classifier(con.to(self._device))
            sim = sim.cpu()
            sims.append(sim)
        sims = torch.cat(sims, dim=0)

        return sims

    @torch.no_grad()
    def avg_non_zero_concept_ratio(self, 
                                   dataset: Dataset, 
                                   saved_activation_path: str, 
                                   data_label: str) -> torch.Tensor:
        '''
        Get the average amount of non zero values in the concept bottleneck. 

        Parameters
        ----------
        dataset: Dataset
            The dataset for which the concept similarities should be 
            computed for. 
        saved_activation_path: str
            Path where the concept similarities can be/are saved. 
        data_label: str
            Label (train or test) from given dataset. 

        Returns
        -------
        non_zero_ratio: float
        '''

        self._classifier.eval()

        # Load the concept activations. 
        embeddings = self._get_concept_embeddings(
            dataset, saved_activation_path, data_label, 
            self._normalize, self._mean, self._std)
        
        data_loader = DataLoader(embeddings, batch_size=self._batch_size, 
                                 shuffle=False, num_workers=8)
        sum = 0
        for con in data_loader:
            _, sim, _ = self._classifier(con.to(self._device))
            active = sim != 0
            if self._gated:
                gates = self.get_gate_values() > 0.5
                active = active & gates.unsqueeze(0)
            sum += float(active.sum().cpu() / sim.shape[1])

        return sum / len(embeddings)

    @torch.no_grad()
    def _get_concept_activations(self,
                                 dataset: Dataset,
                                 saved_activation_path: str,
                                 data_label: str) -> torch.Tensor:
        '''
        Get the per-image, per-concept activation, in the same order as
        dataset. This is self._classifier(...)'s second return value,
        which reflects the relu/jumpReLU/TopK selection and, when
        gated=True, the learned gate applied on top of it.

        Returns
        -------
        activations: torch.Tensor
            Shape (len(dataset), #concepts).
        '''

        self._classifier.eval()

        embeddings = self._get_concept_embeddings(
            dataset, saved_activation_path, data_label,
            self._normalize, self._mean, self._std)

        data_loader = DataLoader(embeddings, batch_size=self._batch_size,
                                 shuffle=False, num_workers=8)
        acts = []
        for con in data_loader:
            con = con.to(self._device)
            _, act, _ = self._classifier(con)
            acts.append(act.cpu())

        return torch.cat(acts, dim=0)

    @torch.no_grad()
    def fit_concept_attribute_mapping(self,
                                      dataset: Dataset,
                                      saved_activation_path: str,
                                      data_label: str) -> torch.Tensor:
        '''
        Align every concept to the CUB attribute whose ground-truth
        presence correlates best with the concept's activation on the
        given dataset, e.g. the training split. Needed before calling
        concept_accuracy.

        Parameters
        ----------
        dataset: Dataset
            A CUB2011 instance loaded with load_attributes=True.
        saved_activation_path: str
        data_label: str

        Returns
        -------
        concept_attr_map: torch.Tensor
            Shape (#concepts,), dtype long. concept_attr_map[c] is the
            column index into dataset.attributes of the attribute best
            aligned with concept c.
        '''

        assert hasattr(dataset, "attributes"), \
            "Dataset must be loaded with load_attributes=True. "

        acts = self._get_concept_activations(
            dataset, saved_activation_path, data_label)
        attrs = dataset.attributes.float()

        acts_c = acts - acts.mean(dim=0, keepdim=True)
        attrs_c = attrs - attrs.mean(dim=0, keepdim=True)

        acts_norm = acts_c.norm(dim=0).clamp_min(1e-8)
        attrs_norm = attrs_c.norm(dim=0).clamp_min(1e-8)

        # Pearson correlation between every concept and every attribute,
        # shape (#concepts, #attributes).
        corr = (acts_c.T @ attrs_c) / (acts_norm[:, None] * attrs_norm[None, :])

        concept_attr_score, concept_attr_map = corr.max(dim=1)
        self._concept_attr_map = concept_attr_map
        self._concept_attr_score = concept_attr_score
        return concept_attr_map

    @torch.no_grad()
    def concept_accuracy(self,
                         dataset: Dataset,
                         saved_activation_path: str,
                         data_label: str,
                         activation_threshold: float = 0.0) -> dict:
        '''
        Measure how many of the concepts the model predicts as present in
        an image are actually present, using CUB ground-truth attribute
        labels. Requires fit_concept_attribute_mapping to have been
        called first (typically on the training split, to avoid leakage).

        Parameters
        ----------
        dataset: Dataset
            A CUB2011 instance loaded with load_attributes=True.
        saved_activation_path: str
        data_label: str
        activation_threshold: float = 0.0
            A concept counts as "predicted" for an image if its
            activation exceeds this value. For gated models, only
            concepts with a learned gate > 0.5 are ever eligible to be
            predicted, matching the threshold used elsewhere (e.g.
            avg_non_zero_concept_ratio).

        Returns
        -------
        dict with:
            "concept accuracy": mean, over images with at least one
                predicted concept, of the fraction of that image's
                predicted concepts whose aligned attribute is actually
                present in the image (i.e. mean per-image precision).
            "avg predicted concepts": mean number of predicted concepts
                per image.
            "images with predictions": amount of images with at least
                one predicted concept (denominator of "concept accuracy").
        '''

        assert hasattr(self, "_concept_attr_map"), \
            "Call fit_concept_attribute_mapping first. "
        assert hasattr(dataset, "attributes"), \
            "Dataset must be loaded with load_attributes=True. "

        acts = self._get_concept_activations(
            dataset, saved_activation_path, data_label)
        attrs = dataset.attributes.float()

        predicted = acts > activation_threshold
        if self._gated:
            gates = self.get_gate_values().cpu() > 0.5
            predicted = predicted & gates.unsqueeze(0)

        attr_hit = attrs[:, self._concept_attr_map.cpu()]  # (n, #concepts)
        correct = predicted & (attr_hit > 0.5)

        n_predicted = predicted.sum(dim=1)
        n_correct = correct.sum(dim=1)

        has_pred = n_predicted > 0
        precision = torch.zeros(acts.shape[0])
        precision[has_pred] = \
            n_correct[has_pred].float() / n_predicted[has_pred].float()

        return {
            "concept accuracy":
                precision[has_pred].mean().item() if has_pred.any()
                else float("nan"),
            "avg predicted concepts": n_predicted.float().mean().item(),
            "images with predictions": int(has_pred.sum().item()),
        }

    def save_to_file(self, filepath: str, filename: str):
        '''
        Saves the classifier of the model into a file. 

        Parameters
        ----------
        filepath: str
            Filepath to save the model. 
        filename: str
            Filename for the file. 
        '''

        def get_backbone():
            try:
                return self._backbone.cpu()
            except AttributeError:
                return None

        path = os.path.join(filepath, filename)
        torch.save(
            {
                "model_state_dict": self._classifier.state_dict(),
                "backbone": get_backbone(),
                "epochs": self._epochs,
                "batch_size": self._batch_size,
                "lam_pi": self._lam_pi,
                "lambda_gate": self._lambda_gate,
                "lam_w": self._lam_w,
                "dropout_p": self._dropout_p, 
                "num_concepts": self._num_concepts, 
                "num_classes": self._num_classes, 
                "w": self._h.detach().cpu(), 
                "learning_rate": self._lr, 
                "relu": self._relu, 
                "scale_mode": self._scale_mode, 
                "bias_mode": self._bias_mode, 
                "multilabel": self._multilabel, 
                "normalize": self._normalize, 
                "mean": self._mean, 
                "std": self._std, 
                "k": self._k,
                "gated": self._gated,
                # "gate_temperature"/"gate_temperature_final" are the configured
                # start/end of the anneal (None end = fixed temperature); the
                # classifier's actual current temperature -- what fit() last
                # set it to, i.e. the annealed-to value at convergence -- is
                # saved separately as "gate_temperature_used" so a reload
                # reproduces eval-time behavior exactly rather than
                # reconstructing the classifier at the *starting* temperature.
                "gate_temperature": self._gate_temperature,
                "gate_temperature_final": self._gate_temperature_final,
                "gate_temperature_used": self._classifier.gate_temperature if self._gated else None,
                "gate_forward": self._gate_forward,
            }, path)
    
    @classmethod
    def load_from_file(cls, 
                       filepath: str, 
                       filename: str, 
                       device: Literal["cuda", "cpu"] = "cuda", 
                       backbone_p = None):
        '''
        Load the classifier of the model from a file. 

        Parameters
        ----------
        filepath: str
            Filepath to save the model. 
        filename: str
            Filename for the file. 
        device: Literal["cuda", "cpu"]
        backbone_p = None
            The backbone if backbone is function (can't be saved). 
        '''

        path = os.path.join(filepath, filename)
        data: dict = torch.load(path)
        if data["backbone"] is not None:
            backbone = data["backbone"].to(device)
        elif backbone_p is not None:
            backbone = backbone_p
        else:
            raise AttributeError()
        h = data["w"].to(device)
        num_concepts = data["num_concepts"]
        num_classes = data["num_classes"]
        # "lam_pi"/"lambda_gate" are the current key names; "lam_gate"/
        # "gated_probe_lam" are what older checkpoints on disk were saved
        # under (same quantities, old names -- lam_gate meant lam_pi, and
        # gated_probe_lam meant lambda_gate).
        lam_pi = data.get("lam_pi", data.get("lam_gate"))
        lambda_gate = data.get("lambda_gate", data.get("gated_probe_lam", lam_pi))
        lam_w = data["lam_w"]
        batch_size = data["batch_size"]
        learning_rate = data["learning_rate"]
        epochs = data["epochs"]
        relu = data.get("relu", "ReLU")
        scale_mode = data["scale_mode"]
        bias_mode = data["bias_mode"]
        scale = data.get("scale", None)
        bias = data.get("bias", None)
        if torch.is_tensor(scale) and \
            torch.allclose(scale, torch.ones(num_concepts).to(scale.device)):
            scale_mode = "no"
        elif scale is None:
            pass
        else:
            raise NotImplementedError
        if torch.is_tensor(bias) and \
            torch.allclose(bias, torch.zeros(num_concepts).to(bias.device)):
            bias_mode = "no"
        elif bias is None:
            pass
        else:
            raise NotImplementedError
        dropout_p = data["dropout_p"]
        multilabel = data.get("multilabel", False)
        normalize = data.get("normalize", False)
        mean = data.get("mean", None)
        std = data.get("std", None)
        k = data.get("k", -1)
        gated = data.get("gated", False)
        gate_temperature = data.get("gate_temperature", 1.0)
        gate_temperature_final = data.get("gate_temperature_final", None)
        # The classifier is reconstructed at the temperature it actually
        # converged to (what fit() last set it to), not the configured
        # starting temperature -- otherwise reloading an annealed model
        # would reset it back to the start of the schedule and silently
        # change its eval-time gate behavior. Older checkpoints saved before
        # annealing existed have no "gate_temperature_used" key; they were
        # always fixed at "gate_temperature", so that's the right fallback.
        gate_temperature_used = data.get("gate_temperature_used", gate_temperature)
        gate_forward = data.get("gate_forward", "soft")  # checkpoints predating the flag were soft
        if scale_mode == "no" and bias_mode == "no" and k == -1 and "relu" not in data:
            relu = "no"

        classifier = Classifier(
            num_concepts,
            num_classes,
            relu,
            scale_mode,
            bias_mode,
            dropout_p,
            k,
            gated=gated,
            gate_temperature=(gate_temperature_used if gate_temperature_used is not None else gate_temperature),
            gate_forward=gate_forward,
        )
        if "top_k.k" in data["model_state_dict"]:
            del data["model_state_dict"]["top_k.k"]
        classifier.load_state_dict(data["model_state_dict"])
        classifier = classifier.eval().to(device)

        ucbm = UCBM(
            backbone,
            h,
            batch_size,
            epochs,
            lam_pi,
            lambda_gate,
            lam_w,
            dropout_p,
            learning_rate,
            relu,
            scale_mode,
            bias_mode,
            normalize,
            k,
            device,
            gated,
            gate_temperature,
            gate_temperature_final,
            gate_forward,
        )
        ucbm._classifier = classifier
        ucbm._num_classes = num_classes
        ucbm._multilabel = multilabel
        ucbm._mean = mean
        ucbm._std = std
        return ucbm

    @torch.no_grad()
    def compute_confusion_matrix(self, 
                                 dataset: ImageFolder, 
                                 saved_activation_path: str, 
                                 data_label: str) \
        -> dict[int, dict[int, float]]:
        '''
        Function that computes the confusion matrix for this model. 

        Parameters
        ----------
        dataset: ImageFolder
            The dataset for which the concept similarities should be 
            computed for. 
        saved_activation_path: str
            Path where the concept similarities can be/are saved. 
        data_label: str
            Label (train or test) from given dataset. 
        
        Returns
        -------
        confusion_matrix: dict[int, dict[int, float]]
            class_id: {other_class: percentage of class_id images
                                    mapped on other class}
        '''

        assert not self._multilabel

        self._classifier.eval()

        # Load the concept activations. 
        embeddings = self._get_concept_embeddings(
            dataset, saved_activation_path, data_label, 
            self._normalize, self._mean, self._std)
        
        classes = dataset.class_to_idx.values()
        confusion_matrix = {c1: {c2: 0 for c2 in classes} for c1 in classes}
        
        dset = PDataset(embeddings, dataset.targets)
        data_loader = DataLoader(dset, batch_size=self._batch_size, 
                                 shuffle=False, num_workers=8)
        
        for X_batch, y_batch in data_loader:
            y_pred, _, _ = self._classifier(X_batch.to(self._device))
            y_pred = y_pred.cpu()
            for i in range(y_pred.numel()):
                confusion_matrix[y_batch[i]][y_pred[i]] += 1
        
        all = len(embeddings)
        confusion_matrix = {c: {k: v / all for k, v in cv.items()} 
                              for c, cv in confusion_matrix.items()}
        
        return confusion_matrix
    
    def get_classifier_weights(self) -> torch.Tensor:
        '''
        Get weights of the classifier. 

        Returns
        -------
        weights: torch.Tensor
            Shape (#concepts, #classes)
        '''

        return self._classifier.linear.weight
    
    def get_classifier_bias(self) -> torch.Tensor:
        '''
        Get bias of the classifier. 

        Returns
        -------
        bias: torch.Tensor
            Shape (#classes)
        '''

        return self._classifier.linear.bias

    def get_gate_values(self) -> torch.Tensor:
        '''
        Get the learned gate values of the classifier. Only valid if this
        model was fit with gated=True.

        Returns
        -------
        gates: torch.Tensor
            Shape (#concepts). sigmoid(gate_logits / gate_temperature),
            each in [0, 1].
        '''

        assert self._gated, "Model was not fit with gated=True. "
        return self._classifier.gate_probs().detach()

    def prune_and_refit(self, training_data, test_data, saved_activation_path, tau=0.5,
                        epochs=20, lr=1e-3, seed=0):
        """Mask-and-refit of a gated classifier: K = gate > tau, linear head refit on the
        post-selection (pre-gate) concept signal restricted to K. Returns a dict with the
        refit head (`W_refit` applies to the pre-gate signal, no gate at inference), the
        concept set, and test accuracy of the hard cut with and without the refit."""
        assert self._gated, "prune_and_refit needs a gated classifier"
        gate = self.get_gate_values().detach().cpu().float().clamp_min(1e-12)
        # _get_concept_activations returns the post-gate signal; divide the gate back out.
        pre_train = self._get_concept_activations(training_data, saved_activation_path, "train") / gate
        pre_test = self._get_concept_activations(test_data, saved_activation_path, "test") / gate
        y_train = torch.tensor(training_data.targets)
        y_test = torch.tensor(test_data.targets)
        W, b = self._classifier.linear.weight.detach(), self._classifier.linear.bias.detach()
        masked_acc = _masked_accuracy(W, b, gate, pre_test, y_test, tau=tau, fold_gate=True,
                                      device=self._device)
        W_r, b_r, keep = _prune_and_refit(W, b, gate, pre_train, y_train, tau=tau, fold_gate=True,
                                          epochs=epochs, lr=lr, batch_size=self._batch_size,
                                          device=self._device, seed=seed)
        acc = _evaluate_head(W_r, b_r, pre_test, y_test, device=self._device)
        return {"tau": tau, "n_concepts": int(keep.sum()), "masked_accuracy_no_refit": masked_acc,
                "test_accuracy": acc, "W_refit": W_r, "b_refit": b_r, "keep": keep}

    def get_info_dict(self,
                      training_data: ImageFolder, 
                      test_data: ImageFolder, 
                      saved_activation_path: str,
                      metrics = ["acc", "auprc", "auprc_pc", "auroc"]) -> dict:
        '''
        Get the most important information abou this dict in a dictionary. 

        Parameters
        ----------
        training_data: ImageFolder
        test_data: ImageFolder
            Data used to compute test accuracy, ...
        saved_activation_path: str
        '''

        data = dict()
        if self._gated:
            active_gates = self.get_gate_values() > 0.5
            data["amount of concepts"] = int(active_gates.sum().item())
        else:
            # Not just self._h.shape[0] -- that's the full dictionary size and
            # ignores whatever elastic_loss_weights/lam_w actually pruned. A
            # concept counts as used if any class has a nonzero weight on it,
            # same criterion as the gated case's "affects at least one class".
            weight = self._classifier.linear.weight.data
            data["amount of concepts"] = int((weight.abs() > 1e-5).any(dim=0).sum().item())
        data["gated"] = self._gated

        # "amount of concepts" above is a single thresholded count (gate>0.5, or
        # nonzero-weight union) and is easy to misread -- e.g. a gated run whose
        # gates all fell below 0.5 reads as 0 concepts at unchanged accuracy.
        # "concept_usage" is the backbone-agnostic panel: contribution-mass
        # effective counts, participation ratio, per-image active concepts, NEC.
        W = self._classifier.linear.weight.data
        acts = self._get_concept_activations(test_data, saved_activation_path, "test")
        if self._gated:
            # acts already have the gate folded in (2nd forward output); recover the
            # pre-gate code and hand the gate over separately so every metric -- not
            # just the contribution terms -- sees it.
            gate = self.get_gate_values().detach().cpu()
            data["concept_usage"] = concept_usage_report(
                W, acts / gate.clamp_min(1e-12), gate=gate)
        else:
            data["concept_usage"] = concept_usage_report(W, acts)

        data["amount of classes"] = len(test_data.classes)
        train_res = self.get_evaluation_metric(
            training_data, metrics, saved_activation_path, "train")
        test_res = self.get_evaluation_metric(
            test_data, metrics, saved_activation_path, "test")
        if "acc" in metrics and "acc" in train_res:
            data["train acc"] = train_res["acc"]
            data["test acc"] = test_res["acc"]
        if "auprc" in metrics and "auprc" in train_res:
            data["train auprc"] = train_res["auprc"]
            data["test auprc"] = test_res["auprc"]
        if "auprc_pc" in metrics and "auprc_pc" in train_res:
            data["train auprc_pc"] = train_res["auprc_pc"]
            data["test auprc_pc"] = test_res["auprc_pc"] 
        if "auroc" in metrics and "auroc" in train_res:
            data["train auroc"] = train_res["auroc"]
            data["test auroc"] = test_res["auroc"]

        data["avg non zero concept ratio"] = \
            self.avg_non_zero_concept_ratio(
                test_data, saved_activation_path, "test")
        data["learning rate"] = self._lr
        data["lambda pi"] = self._lam_pi
        data["lambda gate"] = self._lambda_gate
        data["gate temperature start"] = self._gate_temperature
        data["gate temperature final"] = self._gate_temperature_final
        data["gate forward"] = self._gate_forward
        # Actual temperature the classifier converged to and is currently
        # using for eval -- equals "gate temperature start" when
        # gate_temperature_final is None (fixed, non-annealed temperature).
        data["gate temperature"] = self._classifier.gate_temperature if self._gated else self._gate_temperature
        data["lambda w"] = self._lam_w
        data["epochs"] = self._epochs
        data["dropout p"] = self._dropout_p
        data["scale mode"] = self._scale_mode
        data["bias mode"] = self._bias_mode
        data["multilabel"] = self._multilabel
        data["normalize"] = self._normalize
        data["k"] = self._k
    
        return data