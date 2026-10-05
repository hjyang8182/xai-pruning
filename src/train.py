import torch
import torch.nn as nn
from src.config import LEARNING_RATE, EPOCHS, LAMBDA_SPARSE, LAMBDA_GATE, PROBE_BATCH_SIZE, device
from src.models import LinearProbe
from gating import (train_gated_probe as gated_train_gated_probe,
                    resolve_store_device, stage_acts, iter_minibatches)


def _predict(probe, acts, y, project, autocast, batch_size, device):
    """Chunked argmax over `acts`, applying `project` per chunk. Chunking matters when
    `project` expands the input (a k-dim concept layer on top of CLIP features): the full
    projected eval matrix can be far larger than the features it comes from."""
    correct = 0
    with torch.no_grad():
        for i in range(0, len(acts), batch_size):
            batch = acts[i:i + batch_size].to(device, non_blocking=True).float()
            with autocast:
                if project is not None:
                    batch = project(batch)
                preds = probe(batch).argmax(dim=1)
            correct += (preds == y[i:i + batch_size].to(device)).sum().item()
    return correct / len(acts)


def train_probes_shared_projection(train_feats, y_train, test_feats, y_test, project, n_concepts_list,
                                   n_classes=100, lambda_sparse=LAMBDA_SPARSE, lr=LEARNING_RATE,
                                   epochs=EPOCHS, batch_size=PROBE_BATCH_SIZE, device=device,
                                   acts_device="auto", amp=False, seed=None):
    """Train one probe per entry of `n_concepts_list`, all sharing a single projection matmul.

    Only valid when the smaller concept layers are column-prefixes of the widest one --
    which holds for RandomConceptLayer, whose weight rows come off a seeded generator in
    order. `project` produces the widest activations; probe j sees `[:, :n_concepts_list[j]]`.

    One projection per minibatch instead of one per concept count: on places365 that halves
    the run and takes GPU utilization from 82% to 97% (fp32). The tradeoff is that every
    probe now walks the same shuffle order, so a given (seed, count) no longer reproduces
    the sequential `train_probe` path exactly -- statistically equivalent, not identical.
    """
    probes, optimizers = [], []
    for k in n_concepts_list:
        if seed is not None:
            torch.manual_seed(seed)  # match the per-probe init the sequential path draws
        probe = LinearProbe(k, n_classes).to(device)
        probes.append(probe)
        optimizers.append(torch.optim.Adam(probe.parameters(), lr=lr))
    criterion = nn.CrossEntropyLoss()

    store_device = resolve_store_device((train_feats, test_feats), device, acts_device)
    train_feats, y_train = stage_acts(train_feats, y_train, store_device)
    test_feats, y_test = stage_acts(test_feats, y_test, store_device)
    autocast = torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp)

    for _ in range(epochs):
        for feats_batch, labels_batch in iter_minibatches(train_feats, y_train, batch_size):
            feats_batch = feats_batch.to(device, non_blocking=True).float()
            labels_batch = labels_batch.to(device, non_blocking=True)
            with autocast:
                with torch.no_grad():
                    acts = project(feats_batch)
                for probe, optimizer, k in zip(probes, optimizers, n_concepts_list):
                    loss = (criterion(probe(acts[:, :k]), labels_batch)
                            + lambda_sparse * probe.linear.weight.abs().sum())
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

    accs = [0] * len(probes)
    for probe in probes:
        probe.eval()
    with torch.no_grad():
        for i in range(0, len(test_feats), batch_size):
            batch = test_feats[i:i + batch_size].to(device, non_blocking=True).float()
            labels = y_test[i:i + batch_size].to(device)
            with autocast:
                acts = project(batch)
                for j, (probe, k) in enumerate(zip(probes, n_concepts_list)):
                    accs[j] += (probe(acts[:, :k]).argmax(dim=1) == labels).sum().item()
    return [(a / len(test_feats), p) for a, p in zip(accs, probes)]


def train_probe(train_acts, y_train, test_acts, y_test,
                n_classes=100, lambda_sparse=LAMBDA_SPARSE, lr=LEARNING_RATE,
                epochs=EPOCHS, batch_size=PROBE_BATCH_SIZE, device=device, track_history=False,
                acts_device="auto", amp=False, project=None, n_concepts=None):
    """`project` (with `n_concepts`) trains on a frozen transform of `train_acts` computed
    per minibatch instead of on the activations themselves -- pass the CLIP features plus a
    frozen concept layer and the (n_examples, n_concepts) matrix is never materialized. For
    places365 that matrix is ~59GB at 8192 concepts, so materializing it means streaming it
    from host memory every epoch; projecting on the fly costs one extra matmul per batch."""
    if project is not None and n_concepts is None:
        raise ValueError("train_probe(project=...) also needs n_concepts (the projection's output dim)")
    probe = LinearProbe(n_concepts if project is not None else train_acts.shape[1], n_classes).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Keep the fixed activation matrices resident (GPU if they fit, else pinned CPU) and
    # slice a fresh permutation each epoch -- replaces DataLoader(TensorDataset), whose
    # per-batch collate + host->device copy dominates once the model is a single Linear.
    store_device = resolve_store_device((train_acts, test_acts), device, acts_device)
    train_acts, y_train = stage_acts(train_acts, y_train, store_device)
    test_acts, y_test = stage_acts(test_acts, y_test, store_device)
    autocast = torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp)

    history = {'train_loss': [], 'test_acc': []} if track_history else None
    for _ in range(epochs):
        probe.train()
        epoch_loss, n_batches = 0.0, 0
        for acts_batch, labels_batch in iter_minibatches(train_acts, y_train, batch_size):
            acts_batch = acts_batch.to(device, non_blocking=True).float()
            labels_batch = labels_batch.to(device, non_blocking=True)
            with autocast:
                if project is not None:
                    with torch.no_grad():
                        acts_batch = project(acts_batch)
                sparsity_loss = probe.linear.weight.abs().sum()
                loss = criterion(probe(acts_batch), labels_batch) + lambda_sparse * sparsity_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if track_history:
                epoch_loss += loss.item()
                n_batches += 1
        if track_history:
            probe.eval()
            history['train_loss'].append(epoch_loss / n_batches)
            history['test_acc'].append(
                _predict(probe, test_acts, y_test, project, autocast, batch_size, device))
    probe.eval()
    acc = _predict(probe, test_acts, y_test, project, autocast, batch_size, device)
    if track_history:
        return acc, probe, history
    return acc, probe


def train_gated_probe(train_acts, y_train, test_acts, y_test,
                    n_classes=100, lambda_sparse=LAMBDA_SPARSE, lambda_gate=LAMBDA_GATE,
                    lr=LEARNING_RATE, epochs=EPOCHS, batch_size=PROBE_BATCH_SIZE, device=device,
                    track_history=False, acts_device="auto", amp=False, gate_forward="soft",
                    gate_init=2.0, gate_init_std=0.0):
    # Delegates to the backbone-agnostic implementation in gating.py (shared with any other CBM
    # backbone that trains a GatedProbe), plugging in DN-CBM's own config defaults.
    return gated_train_gated_probe(
        train_acts, y_train, test_acts, y_test, n_classes,
        lr=lr, epochs=epochs, batch_size=batch_size,
        lambda_sparse=lambda_sparse, lambda_gate=lambda_gate,
        device=device, track_history=track_history,
        acts_device=acts_device, amp=amp, gate_forward=gate_forward,
        gate_init=gate_init, gate_init_std=gate_init_std,
    )

def get_predictions(probe, acts, device='cpu'):
    probe = probe.to(device).eval()
    with torch.no_grad():
        return probe(acts.to(device)).argmax(dim=1).cpu()

def get_gated_predictions(probe, acts, device='cpu'):
    probe = probe.to(device).eval()
    with torch.no_grad():
        logits, _ = probe(acts.to(device))
        return logits.argmax(dim=1).cpu()

def concept_contributions(probe, acts, device=device):
    """Each concept's contribution to the predicted class's logit: effective_acts *
    probe.linear.weight[pred], where effective_acts is acts * gates for a GatedProbe (same
    definition used by explain_image for single-example explanations). Returns (pred,
    contributions), both batched over acts's leading dimension."""
    probe = probe.to(device).eval()
    acts = acts.to(device)
    with torch.no_grad():
        out = probe(acts)
        if isinstance(out, tuple):
            logits, gates = out
            effective_acts = acts * gates
        else:
            logits = out
            effective_acts = acts
        pred = logits.argmax(dim=1)
        contributions = effective_acts * probe.linear.weight[pred]
    return pred, contributions


def topk_prediction_change_rate(probe, acts, top_k, device=device):
    """Fraction of predictions that flip when activations are masked to keep only the top_k
    concepts contributing to the original predicted class. Measures how faithful the probe's
    decision is to its own top-k explanation."""
    pred, contributions = concept_contributions(probe, acts, device=device)
    acts = acts.to(device)
    with torch.no_grad():
        k = min(top_k, contributions.shape[1])
        top_idx = contributions.abs().topk(k, dim=1).indices
        mask = torch.zeros_like(acts)
        mask.scatter_(1, top_idx, 1.0)

        pruned_out = probe(acts * mask)
        pruned_logits = pruned_out[0] if isinstance(pruned_out, tuple) else pruned_out
        pruned_pred = pruned_logits.argmax(dim=1)
        return (pruned_pred != pred).float().mean().item()


def group_accuracy(preds, labels, fine_to_coarse):
    """Superclass (group) accuracy: fraction of predictions whose mapped coarse
    label matches the true coarse label, given a fine-label -> coarse-label mapping fn."""
    coarse_preds = fine_to_coarse(preds)
    coarse_labels = fine_to_coarse(labels)
    return (coarse_preds == coarse_labels).float().mean().item()
