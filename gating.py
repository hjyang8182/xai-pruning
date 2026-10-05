"""Backbone-agnostic gated concept probe.

A linear classifier over concept activations with a learned per-concept gate, plus the generic
train/eval loop for it. Everything here operates on plain concept-activation tensors (n_examples x
n_concepts) and integer labels, so it drops in for any CBM backbone - DN-CBM (src/), LF-CBM, UCBM,
or anything else that ends up producing a concept-activation matrix - regardless of how those
activations were computed upstream. No dependency on any one backbone's config or data pipeline.

DN-CBM (src/models.py, src/train.py) imports directly from here. Backbones that train gates as
part of a larger joint objective (e.g. UCBM's classifier head, which layers gating on top of
several other concept-selection options) can still reuse `GatedProbe` and `gate_sparsity_loss` /
`weight_sparsity_loss` as building blocks without adopting the full `train_gated_probe` loop.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedProbe(nn.Module):
    """Linear head with a learned per-concept gate.

    gate_forward : "soft" -- forward uses the sigmoid value, linear(x * sigmoid(logits/T)).
                             The trained model is *not* the hard-thresholded one: the
                             head can absorb any gate scale ((g, w) -> (g/c, c*w) leaves
                             the logits unchanged), so gates drift toward 0 without ever
                             closing anything and "open @0.5" describes a different model.
                   "hard" -- forward uses the hard mask 1[sigmoid > 0.5], with the
                             sigmoid's gradient passed straight through (STE). The model
                             that is trained is exactly the masked model: a gate below
                             0.5 contributes nothing, so lowering a useful gate costs
                             accuracy at once and the head cannot compensate for a zero.
                             sum(sigmoid) is then a relaxation of the real open count.
    The gate penalty in the training loops always uses the soft sigmoid (it needs a
    gradient); `gate_probs()` is soft in both modes; `forward` returns the soft gates
    as its second output for that reason.
    """
    def __init__(self, n_concepts, n_classes, gate_temperature=1.0, gate_forward="soft",
                 gate_init=2.0, gate_init_std=0.0):
        super().__init__()
        if gate_forward not in ("soft", "hard"):
            raise ValueError(f"gate_forward must be 'soft' or 'hard', got {gate_forward!r}")
        # Default +2.0 -> sigmoid ~= 0.88, so all concepts start approximately open. `gate_init`
        # is the constant every logit starts at (< 0 starts everything closed; with the hard
        # forward the STE still lets closed gates open); `gate_init_std` > 0 adds N(0, std)
        # noise so different concepts start at different gate values (uses the global RNG, so
        # seed before constructing).
        init = torch.full((n_concepts,), float(gate_init))
        if gate_init_std > 0:
            init = init + gate_init_std * torch.randn(n_concepts)
        self.gate_logits = nn.Parameter(init)
        self.linear = nn.Linear(n_concepts, n_classes)
        self.gate_temperature = gate_temperature
        # Stored as a buffer so a checkpoint remembers which forward it was trained with;
        # checkpoints from before this flag existed load as "soft" (see _load_from_state_dict).
        self.register_buffer("hard_forward", torch.tensor(gate_forward == "hard"))

    @property
    def gate_forward(self):
        return "hard" if bool(self.hard_forward) else "soft"

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        if prefix + "hard_forward" not in state_dict:
            state_dict[prefix + "hard_forward"] = torch.tensor(False)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def gate_probs(self):
        """sigmoid(gate_logits / gate_temperature). T < 1 sharpens the gate
        toward 0/1 (more bimodal) for the same spread of logits; T = 1 is
        the plain sigmoid, unaffected. The open/closed threshold at 0.5 is
        unaffected by T since sigmoid(0) = 0.5 regardless."""
        return torch.sigmoid(self.gate_logits / self.gate_temperature)

    def gate_mask(self):
        """The gate as applied in the forward pass: hard 1[g > 0.5] with a straight-through
        sigmoid gradient when gate_forward="hard", the soft sigmoid otherwise."""
        gates = self.gate_probs()
        if self.hard_forward:
            return hard_gate_ste(gates)
        return gates

    def forward(self, x):
        gates = self.gate_probs()
        applied = hard_gate_ste(gates) if self.hard_forward else gates
        return self.linear(x * applied), gates


def hard_gate_ste(gates, threshold=0.5):
    """1[gates > threshold] in the forward pass, d/dgates = 1 in the backward pass
    (straight-through estimator). `gates` must be the soft sigmoid values."""
    hard = (gates > threshold).to(gates.dtype)
    return hard + gates - gates.detach()


def gate_temperature_schedule(epoch, epochs, start, end=None):
    """Geometric (log-linear) decay of the gate's sigmoid temperature from
    `start` to `end` across `epochs` epochs, evaluated at `epoch`
    (0-indexed). Geometric rather than linear so the temperature spends
    proportionally similar time at each order of magnitude, rather than
    the fast part of the decay being crammed into the first few epochs
    when start and end differ by an order of magnitude or more.

    Returns `start` unchanged if `end` is None or `epochs <= 1`, i.e. a
    fixed (non-annealed) temperature throughout -- this is the previous
    fixed-gate_temperature behavior, so leaving `end=None` reproduces it
    exactly.
    """
    if end is None or epochs <= 1:
        return start
    frac = epoch / (epochs - 1)
    return start * (end / start) ** frac


def resolve_penalty_temp(mode, forward_temp, gate_temperature, gate_temperature_final):
    """Which sigmoid temperature the gate *sparsity penalty* uses, independent of the
    temperature the forward gate uses.

    The penalty sum(sigmoid(g / T)) is a monotone surrogate for the open-gate count
    sum(1[g >= 0]) at any T (the fold threshold g = 0 is T-invariant), but its gradient
    d/dg sigmoid(g / T) scales as 1/T, so when gate_temperature is annealed downward the
    effective sparsity pressure ramps up over training even at fixed lambda_gate.

    - "forward": track the live (possibly annealed) forward temperature -- couples the two,
      previous behavior.
    - "final": use the annealing endpoint (gate_temperature_final, or gate_temperature if not
      annealing) -- a fixed L0 surrogate as sharp as where the folded model is evaluated.
    - a number (e.g. 1.0): that fixed temperature -- fully decouples the penalty from T so
      lambda_gate keeps a stable meaning across T settings and across an anneal schedule.
    """
    if mode == "forward" or mode is None:
        return forward_temp
    if mode == "final":
        return gate_temperature_final if gate_temperature_final is not None else gate_temperature
    return float(mode)


def resolve_store_device(tensors, device, mode="auto", headroom=0.6):
    """Where should the (fixed) concept-activation tensors live during probe training?

    Probe training is thousands of passes over the same activation matrix, so the
    DataLoader/TensorDataset path (CPU collate + a host->device copy every minibatch) is
    pure overhead. If the tensors fit in GPU memory we keep them resident and slice on the
    GPU; otherwise we keep them in pinned host memory for async per-batch transfer.

    mode: "cuda" / "cpu" force it; "auto" keeps them on `device` when they fit within
    `headroom` of free GPU memory, else falls back to pinned CPU.
    Returns the torch.device to store them on.
    """
    if device.type != "cuda" or mode == "cpu":
        return torch.device("cpu")
    if mode == "cuda":
        return device
    nbytes = sum(t.element_size() * t.nelement() for t in tensors)
    try:
        free, _ = torch.cuda.mem_get_info(device)
    except Exception:
        return torch.device("cpu")
    return device if nbytes < headroom * free else torch.device("cpu")


def stage_acts(acts, labels, store_device):
    """Move an activation/label pair onto `store_device` once; pin if it stays on CPU so the
    per-batch `.to(device, non_blocking=True)` in the training loop overlaps with compute."""
    acts = acts.to(store_device)
    labels = labels.to(store_device)
    if store_device.type == "cpu":
        acts, labels = acts.pin_memory(), labels.pin_memory()
    return acts, labels


def iter_minibatches(acts, labels, batch_size):
    """Yield (acts[idx], labels[idx]) for a fresh random permutation each call (drawn from
    the global RNG, so seed with torch.manual_seed upstream -- same contract as the
    DataLoader(shuffle=True) this replaces). Indexing happens on whatever device `acts` is
    on; the caller moves the batch to the compute device (a no-op when already resident)."""
    n = acts.shape[0]
    perm = torch.randperm(n, device=acts.device)
    for i in range(0, n, batch_size):
        idx = perm[i:i + batch_size]
        yield acts[idx], labels[idx]


def gate_sparsity_loss(gates):
    return gates.sum()


def weight_sparsity_loss(weight):
    return F.l1_loss(weight, torch.zeros_like(weight))


def open_gate_count(probe, threshold=0.5):
    return (probe.gate_probs() > threshold).sum().item()


def get_gated_predictions(probe, acts, device='cpu'):
    probe = probe.to(device).eval()
    with torch.no_grad():
        logits, _ = probe(acts.to(device))
        return logits.argmax(dim=1).cpu()


def train_gated_probe(train_acts, y_train, test_acts, y_test, n_classes,
                       lr=1e-3, epochs=100, batch_size=64,
                       lambda_sparse=1e-4, lambda_gate=1e-4, lambda_gate_warmup=0,
                       gate_temperature=1.0, gate_temperature_final=None,
                       gate_penalty_temp="forward", gate_forward="soft",
                       gate_init=2.0, gate_init_std=0.0,
                       early_stop_patience=0, early_stop_delta=0.5,
                       device=None, track_history=False,
                       acts_device="auto", amp=False):
    """`gate_init` / `gate_init_std`: initial gate logits (see GatedProbe).

    `lambda_gate_warmup > 0` linearly ramps the gate penalty from 0 to `lambda_gate` over the
    first that-many epochs so the linear head settles before concepts get pruned -- without it
    the near-constant sparsity gradient outweighs the still-noisy CE gradient early and closes
    gates before the head has learned which concepts matter. 0 applies the full penalty at once.

    `gate_penalty_temp` picks the sigmoid temperature for the sparsity penalty independently of
    the forward gate temperature (see `resolve_penalty_temp`): "forward" (default) reproduces
    the old behavior of reusing the live forward T; pass 1.0 (or "final") to stop an anneal
    schedule from silently ramping the penalty strength.

    `early_stop_patience > 0` banks the sparsest probe whose test accuracy is still within
    `early_stop_delta` (percentage points) of the best epoch seen, and stops once that many
    consecutive epochs fail to improve it (no new best acc, no sparser gate within tolerance),
    restoring the banked probe. 0 disables it and runs all `epochs`. Same rationale/caveat as
    `train_gated_final` in model/cbm.py."""
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    probe = GatedProbe(train_acts.shape[1], n_classes, gate_temperature=gate_temperature,
                       gate_forward=gate_forward, gate_init=gate_init, gate_init_std=gate_init_std).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Keep the fixed activation matrices resident (on GPU if they fit, else pinned CPU) and
    # slice a fresh permutation each epoch -- replaces DataLoader(TensorDataset), whose
    # per-batch collate + host->device copy dominates at these batch sizes.
    store_device = resolve_store_device((train_acts, test_acts), device, acts_device)
    train_acts, y_train = stage_acts(train_acts, y_train, store_device)
    test_acts, y_test = stage_acts(test_acts, y_test, store_device)
    test_acts_dev, y_test_dev = test_acts.to(device).float(), y_test.to(device)
    autocast = torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp)

    history = {'train_loss': [], 'test_acc': [], 'open_gates': []} if track_history else None
    best_state, best_acc, epochs_since_improve = None, -1.0, 0
    for epoch in range(epochs):
        probe.train()
        probe.gate_temperature = gate_temperature_schedule(epoch, epochs, gate_temperature, gate_temperature_final)
        eff_lambda_gate = lambda_gate * (
            1.0 if lambda_gate_warmup <= 0 else min(1.0, (epoch + 1) / lambda_gate_warmup))
        penalty_temp = resolve_penalty_temp(
            gate_penalty_temp, probe.gate_temperature, gate_temperature, gate_temperature_final)
        epoch_loss, n_batches = 0.0, 0
        for acts_batch, labels_batch in iter_minibatches(train_acts, y_train, batch_size):
            acts_batch = acts_batch.to(device, non_blocking=True).float()
            labels_batch = labels_batch.to(device, non_blocking=True)
            with autocast:
                logits, _ = probe(acts_batch)
                loss = (criterion(logits, labels_batch)
                        + lambda_sparse * weight_sparsity_loss(probe.linear.weight)
                        + eff_lambda_gate * gate_sparsity_loss(torch.sigmoid(probe.gate_logits / penalty_temp)))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if track_history:
                epoch_loss += loss.item()
                n_batches += 1
        if track_history or early_stop_patience:
            probe.eval()
            with torch.no_grad(), autocast:
                test_logits, _ = probe(test_acts_dev)
            test_acc = (test_logits.argmax(dim=1) == y_test_dev).float().mean().item()
            open_gates = open_gate_count(probe)
        if track_history:
            history['train_loss'].append(epoch_loss / n_batches)
            history['test_acc'].append(test_acc)
            history['open_gates'].append(open_gates)
        if early_stop_patience:
            new_best_acc = test_acc > best_acc
            best_acc = max(best_acc, test_acc)
            within_tol = test_acc >= best_acc - early_stop_delta / 100.0
            sparser = best_state is None or open_gates < best_state['open_gates'] or (
                open_gates == best_state['open_gates'] and test_acc > best_state['acc'])
            if within_tol and sparser:
                best_state = {'sd': {k: v.detach().clone() for k, v in probe.state_dict().items()},
                              'gate_temperature': probe.gate_temperature,
                              'acc': test_acc, 'open_gates': open_gates, 'epoch': epoch}
            if (within_tol and sparser) or new_best_acc:
                epochs_since_improve = 0
            else:
                epochs_since_improve += 1
            if epochs_since_improve >= early_stop_patience:
                print(f'Early stopping at epoch {epoch}: restoring epoch {best_state["epoch"]} '
                      f'(acc {best_state["acc"]*100:.1f}%, open gates {best_state["open_gates"]})')
                break
    if best_state is not None:
        probe.load_state_dict(best_state['sd'])
        probe.gate_temperature = best_state['gate_temperature']
    probe.eval()
    with torch.no_grad(), autocast:
        logits, gates = probe(test_acts_dev)
    acc = (logits.argmax(dim=1) == y_test_dev).float().mean().item()
    n_open = open_gate_count(probe)
    print(f'Accuracy: {acc*100:.1f}%, Open gates: {n_open}/{train_acts.shape[1]}')
    if track_history:
        return acc, probe, history
    return acc, probe


def train_gated_backbone(backbone, x_train, y_train, x_test, y_test, n_classes,
                          lr=1e-3, epochs=100, batch_size=64,
                          lambda_sparse=1e-4, lambda_gate=1e-4, lambda_gate_warmup=0,
                          gate_temperature=1.0, gate_temperature_final=None,
                          gate_penalty_temp="forward", gate_forward="soft",
                          early_stop_patience=0, early_stop_delta=0.5,
                          device=None, track_history=False):
    """Apply gating to `backbone` and train the resulting gated probe.

    `backbone` is any callable mapping raw/upstream input to (n_examples, n_concepts) concept
    activations - e.g. DN-CBM's `get_sae_acts(autoencoder, feats, device)` partially applied,
    LF-CBM's `proj_layer`, or UCBM's concept extractor. It's called once each on `x_train` and
    `x_test` under `torch.no_grad()`, exactly how every backbone in this project is actually used
    today (frozen upstream encoder, only the probe gets trained) - so backbone parameters are
    never touched here. If `backbone` needs internal batching for large inputs, that's its own
    responsibility, same as `get_sae_acts` already batches internally.

    This is the "give me a backbone, gate it, train it" entry point: it just computes concept
    activations once and hands them to `train_gated_probe`, so swapping backbones (DN-CBM /
    LF-CBM / UCBM / anything else) never requires touching the gating or training code.
    """
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with torch.no_grad():
        train_acts = backbone(x_train)
        test_acts = backbone(x_test)
    return train_gated_probe(
        train_acts, y_train, test_acts, y_test, n_classes,
        lr=lr, epochs=epochs, batch_size=batch_size,
        lambda_sparse=lambda_sparse, lambda_gate=lambda_gate, lambda_gate_warmup=lambda_gate_warmup,
        gate_temperature=gate_temperature, gate_temperature_final=gate_temperature_final,
        gate_penalty_temp=gate_penalty_temp, gate_forward=gate_forward,
        early_stop_patience=early_stop_patience, early_stop_delta=early_stop_delta,
        device=device, track_history=track_history,
    )


# ---------------------------------------------------------------------------
# Mask-and-refit: the deliverable of a gated run.
#
# The gated probe is trained soft (every concept enters scaled by its gate), but the
# concept count reported for it is a hard cut (gate > tau). Those are two different
# models. `prune_and_refit` makes the reported one real: fix K = {j : gate_j > tau},
# delete every other concept, and re-train just the linear head on the survivors. The
# result has exactly |K| concepts, no gate at inference, and its accuracy is the
# accuracy *of K*. Warm-started from the gated weights; a linear head on cached
# activations is convex, so it lands in the same place from scratch, just slower.
# ---------------------------------------------------------------------------

def masked_accuracy(weight, bias, gate, acts, labels, *, tau=0.5, fold_gate=True, device=None,
                    batch_size=65536):
    """Accuracy of the hard cut *without* refitting: concepts with gate <= tau zeroed, the
    trained head and surviving gate values kept. The gap between this and the refit
    accuracy is how far the soft gate is from a real mask."""
    device = torch.device(device) if device is not None else weight.device
    W = weight.detach().float()
    if fold_gate:
        W = W * gate.detach().float().to(W.device)[None, :]
    keep = (gate.detach().float() > tau).to(W.device)
    W = (W * keep[None, :]).to(device)
    b = bias.detach().float().to(device)
    correct = 0
    with torch.no_grad():
        for i in range(0, acts.shape[0], batch_size):
            xb = acts[i:i + batch_size].to(device).float()
            correct += (xb @ W.T + b).argmax(1).eq(labels[i:i + batch_size].to(device)).sum().item()
    return correct / acts.shape[0]


def prune_and_refit(weight, bias, gate, train_acts, y_train, *, tau=0.5, fold_gate=True,
                    epochs=20, lr=1e-3, batch_size=4096, weight_decay=0.0,
                    device=None, acts_device="auto", seed=0):
    """Mask the gated head at `tau` and refit the linear layer on the surviving concepts.

    weight    : (n_classes, n_concepts) trained final-layer weight
    bias      : (n_classes,)
    gate      : (n_concepts,) gate values sigmoid(logits / T) (not logits)
    fold_gate : True when `weight` is stored *ungated* (the gate is applied in the
                forward pass -- GatedProbe, UCBM's Classifier), so the warm start is
                weight * gate. False when the gate is already folded into `weight`
                (LF-CBM / VLG-CBM arm C `W_g`).
    train_acts: (n, n_concepts) the exact input of the linear layer, pre-gate (i.e. the
                raw concept activations for a GatedProbe; UCBM's post-selection signal).

    Returns (W, b, keep):
        W    : (n_classes, n_concepts) float32 CPU, exactly zero outside `keep`
        b    : (n_classes,) float32 CPU
        keep : (n_concepts,) bool CPU, the concept set K = gate > tau
    Inference is `acts @ W.T + b` -- no gate.
    """
    device = torch.device(device) if device is not None else (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    gate = gate.detach().float().cpu()
    keep = gate > tau
    W0 = weight.detach().float().cpu()
    if fold_gate:
        W0 = W0 * gate[None, :]
    b0 = bias.detach().float().cpu() if bias is not None else torch.zeros(W0.shape[0])
    n_classes, n_concepts = W0.shape
    keep_idx = keep.nonzero(as_tuple=True)[0]
    W = torch.zeros(n_classes, n_concepts)
    if keep_idx.numel() == 0:  # nothing survives: bias-only model
        return W, b0.clone(), keep

    torch.manual_seed(seed)
    head = nn.Linear(keep_idx.numel(), n_classes).to(device)
    with torch.no_grad():
        head.weight.copy_(W0[:, keep_idx])
        head.bias.copy_(b0)
    opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=weight_decay)
    crit = nn.CrossEntropyLoss()

    acts_k = train_acts[:, keep_idx.to(train_acts.device)].float()
    store = resolve_store_device((acts_k, y_train), device, acts_device)
    acts_k, y_tr = stage_acts(acts_k, y_train, store)
    for _ in range(epochs):
        for xb, yb in iter_minibatches(acts_k, y_tr, batch_size):
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            crit(head(xb), yb).backward()
            opt.step()
    W[:, keep_idx] = head.weight.detach().cpu()
    return W, head.bias.detach().cpu(), keep


def evaluate_head(W, b, acts, labels, *, device=None, batch_size=65536):
    """Top-1 accuracy of a plain linear head `acts @ W.T + b` (what `prune_and_refit` returns)."""
    device = torch.device(device) if device is not None else (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    W, b = W.to(device).float(), b.to(device).float()
    correct = 0
    with torch.no_grad():
        for i in range(0, acts.shape[0], batch_size):
            xb = acts[i:i + batch_size].to(device).float()
            correct += (xb @ W.T + b).argmax(1).eq(labels[i:i + batch_size].to(device)).sum().item()
    return correct / acts.shape[0]
