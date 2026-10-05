import sys
import torch
import os
import random
import statistics
import datetime
import json
import argparse

import utils
import data_utils
import similarity

from glm_saga.elasticnet import IndexedTensorDataset, glm_saga
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from gated_probe import GatedProbe, gate_temperature_schedule, prune_and_refit, masked_accuracy, evaluate_head

"""
Trains either the baseline (GLM-SAGA, default) or the gated linear probe (-g/--gated, with mask-and-refit) the same way
train_cbm.py does, but repeats just the probe-training stage across multiple
seeds -- concept discovery and the projection layer are seed-independent and
expensive, so they're computed once and shared across seeds. Records test
accuracy and number of open concepts for each probe at each seed.
"""

parser = argparse.ArgumentParser(description='Train baseline and gated linear probes across multiple seeds')

parser.add_argument("--dataset", type=str, default="cifar10")
parser.add_argument("--concept_set", type=str, default=None,
                    help="path to concept set name")
parser.add_argument("--backbone", type=str, default="clip_RN50", help="Which pretrained model to use as backbone")
parser.add_argument("--clip_name", type=str, default="ViT-B/16", help="Which CLIP model to use")

parser.add_argument("--device", type=str, default="cuda", help="Which device to use")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size used when saving model/CLIP activations")
parser.add_argument("--saga_batch_size", type=int, default=64, help="Batch size used when fitting final layer")
parser.add_argument("--proj_batch_size", type=int, default=50000, help="Batch size to use when learning projection layer")

parser.add_argument("--feature_layer", type=str, default='layer4',
                    help="Which layer to collect activations from. Should be the name of second to last layer in the model")
parser.add_argument("--activation_dir", type=str, default='saved_activations', help="save location for backbone and CLIP activations")
parser.add_argument("--results_dir", type=str, default='seed_sweep_results', help="where to save the aggregated per-seed results json")
parser.add_argument("--save_shared", type=str, nargs="?", const="", default=None, help="Cache the seed-independent stage (projection W_c, normalised train_c/val_c, labels, concepts) to this dir (no value: results_dir/shared_<dataset>_<stamp>) so the other arm can --load_shared it")
parser.add_argument("--load_shared", type=str, default=None, help="Skip concept discovery + projection and load the shared stage from a --save_shared dir")
parser.add_argument("--save_probes", action="store_true", help="Save every seed's probe under one session dir results_dir/<dataset>_<arm>_seed_sweep_<stamp>/seed<S>/ (W_g.pt, b_g.pt, gate_logits.pt, W_refit.pt ...), shaped like a train_cbm.py run dir so concept_retention.py --load-dir can be pointed at it")
parser.add_argument("-g", "--gated", action="store_true", help="Train the gated probe (arm C, + mask-and-refit) across seeds; omit to train the GLM-SAGA baseline (arm A) instead")
parser.add_argument("--gate_forward", choices=["soft", "hard"], default="soft", help="gated arm: 'soft' multiplies by sigmoid(gate); 'hard' uses the mask 1[sigmoid>0.5] in the forward pass with a straight-through gradient (the trained model is exactly the masked one)")
parser.add_argument("--refit_tau", type=float, default=0.5, help="gated arm: keep concepts with gate > tau, drop the rest, refit the head on the survivors (the reported 'gated_refit' arm)")
parser.add_argument("--refit_epochs", type=int, default=20)
parser.add_argument("--no_refit", action="store_true", help="skip the mask-and-refit stage")
parser.add_argument("--clip_cutoff", type=float, default=0.25, help="concepts with smaller top5 clip activation will be deleted")
parser.add_argument("--proj_steps", type=int, default=1000, help="how many steps to train the projection layer for")
parser.add_argument("--proj_patience", type=int, default=1, help="number of consecutive non-improving val checks before stopping projection training (each check is 50 steps)")
parser.add_argument("--interpretability_cutoff", type=float, default=0.45, help="concepts with smaller similarity to target concept will be deleted")
parser.add_argument("--lam", type=float, default=0.0007, help="Sparsity regularization parameter, higher->more sparse")
parser.add_argument("--n_iters", type=int, default=1000, help="How many iterations to run the final layer solver for")
parser.add_argument("--gate_epochs", type=int, default=20, help="How many epochs to train the gate probe for")
parser.add_argument("--gate_lam", type=float, default=1e-4, help="Sparsity penalty on sum(gates) during gate training")
parser.add_argument("--gate_temperature", type=float, default=1.0, help="Temperature T dividing gate_logits before the sigmoid. T < 1 sharpens the gate toward 0/1 (more bimodal); T = 1 is the plain sigmoid. Starting temperature of the anneal if --gate_temperature_final is also given.")
parser.add_argument("--gate_temperature_final", type=float, default=None, help="If given, anneal the gate temperature geometrically from --gate_temperature down (or up) to this value over training, instead of keeping it fixed.")
parser.add_argument("--print", action='store_true', help="Print all concepts being deleted in the filtering stage")
parser.add_argument("--no_filter", action="store_true", help="Skip CLIP and interpretability filtering; use full concept set (recommended with gated probes)")
parser.add_argument("--seeds", type=int, nargs="+", default=[0, 10, 20, 30, 40], help="Seeds to train each probe type with; results are reported per-seed and averaged")


def build_shared_concept_data(args):
    """Concept discovery + projection layer training, i.e. everything in
    train_cbm.py that doesn't depend on the probe-training seed. Returns the
    normalized per-example concept activations (train_c/val_c) and targets
    that every seed's probe training reuses."""

    similarity_fn = similarity.cos_similarity_cubed_single

    d_train = args.dataset + "_train"
    # CUB only has an official train/test split, no val -- everywhere else calls
    # this eval split "_val", but for cub it's really "_test".
    d_val = args.dataset + ("_test" if args.dataset == "cub" else "_val")

    cls_file = data_utils.LABEL_FILES[args.dataset]
    with open(cls_file, "r") as f:
        classes = f.read().split("\n")

    with open(args.concept_set) as f:
        concepts = f.read().split("\n")

    for d_probe in [d_train, d_val]:
        utils.save_activations(clip_name=args.clip_name, target_name=args.backbone,
                               target_layers=[args.feature_layer], d_probe=d_probe,
                               concept_set=args.concept_set, batch_size=args.batch_size,
                               device=args.device, pool_mode="avg", save_dir=args.activation_dir)

    target_save_name, clip_save_name, text_save_name = utils.get_save_names(args.clip_name, args.backbone,
                                            args.feature_layer, d_train, args.concept_set, "avg", args.activation_dir)
    val_target_save_name, val_clip_save_name, text_save_name = utils.get_save_names(args.clip_name, args.backbone,
                                            args.feature_layer, d_val, args.concept_set, "avg", args.activation_dir)

    with torch.no_grad():
        target_features = torch.load(target_save_name, map_location="cpu").float()

        val_target_features = torch.load(val_target_save_name, map_location="cpu").float()

        image_features = torch.load(clip_save_name, map_location="cpu").float()
        image_features /= torch.norm(image_features, dim=1, keepdim=True)

        val_image_features = torch.load(val_clip_save_name, map_location="cpu").float()
        val_image_features /= torch.norm(val_image_features, dim=1, keepdim=True)

        text_features = torch.load(text_save_name, map_location="cpu").float()
        text_features /= torch.norm(text_features, dim=1, keepdim=True)

        clip_features = image_features @ text_features.T
        val_clip_features = val_image_features @ text_features.T

        del image_features, text_features, val_image_features

    highest = torch.mean(torch.topk(clip_features, dim=0, k=5)[0], dim=0)

    if not args.no_filter:
        if args.print:
            for i, concept in enumerate(concepts):
                if highest[i] <= args.clip_cutoff:
                    print("Deleting {}, CLIP top5:{:.3f}".format(concept, highest[i]))
        concepts = [concepts[i] for i in range(len(concepts)) if highest[i] > args.clip_cutoff]
        clip_mask = highest > args.clip_cutoff
    else:
        clip_mask = torch.ones(len(highest), dtype=torch.bool)

    del clip_features
    with torch.no_grad():
        image_features = torch.load(clip_save_name, map_location="cpu").float()
        image_features /= torch.norm(image_features, dim=1, keepdim=True)

        text_features = torch.load(text_save_name, map_location="cpu").float()[clip_mask]
        text_features /= torch.norm(text_features, dim=1, keepdim=True)

        clip_features = image_features @ text_features.T
        del image_features, text_features

    val_clip_features = val_clip_features[:, clip_mask]

    proj_layer = torch.nn.Linear(in_features=target_features.shape[1], out_features=len(concepts),
                                 bias=False).to(args.device)
    opt = torch.optim.Adam(proj_layer.parameters(), lr=1e-3)

    indices = [ind for ind in range(len(target_features))]

    best_val_loss = float("inf")
    best_step = 0
    best_weights = None
    patience_count = 0
    proj_batch_size = min(args.proj_batch_size, len(target_features))
    for i in range(args.proj_steps):
        batch = torch.LongTensor(random.sample(indices, k=proj_batch_size))
        outs = proj_layer(target_features[batch].to(args.device).detach())
        loss = -similarity_fn(clip_features[batch].to(args.device).detach(), outs)

        loss = torch.mean(loss)
        loss.backward()
        opt.step()
        if i % 50 == 0 or i == args.proj_steps - 1:
            with torch.no_grad():
                val_output = proj_layer(val_target_features.to(args.device).detach())
                val_loss = -similarity_fn(val_clip_features.to(args.device).detach(), val_output)
                val_loss = torch.mean(val_loss)
            if i == 0:
                best_val_loss = val_loss
                best_step = i
                best_weights = proj_layer.weight.clone()
                print("Step:{}, Avg train similarity:{:.4f}, Avg val similarity:{:.4f}".format(best_step, -loss.cpu(),
                                                                                               -best_val_loss.cpu()))
            elif val_loss < best_val_loss:
                best_val_loss = val_loss
                best_step = i
                best_weights = proj_layer.weight.clone()
                patience_count = 0
            else:
                patience_count += 1
                if patience_count >= args.proj_patience:
                    break
        opt.zero_grad()

    proj_layer.load_state_dict({"weight": best_weights})
    print("Best step:{}, Avg val similarity:{:.4f}".format(best_step, -best_val_loss.cpu()))

    with torch.no_grad():
        outs = proj_layer(val_target_features.to(args.device).detach())
        sim = similarity_fn(val_clip_features.to(args.device).detach(), outs)
        interpretable = sim > args.interpretability_cutoff

    if not args.no_filter:
        if args.print:
            for i, concept in enumerate(concepts):
                if sim[i] <= args.interpretability_cutoff:
                    print("Deleting {}, Iterpretability:{:.3f}".format(concept, sim[i]))
        concepts = [concepts[i] for i in range(len(concepts)) if interpretable[i]]
        W_c = proj_layer.weight[interpretable]
    else:
        W_c = proj_layer.weight

    del clip_features, val_clip_features
    proj_layer = torch.nn.Linear(in_features=target_features.shape[1], out_features=len(concepts), bias=False)
    proj_layer.load_state_dict({"weight": W_c})

    train_targets = data_utils.get_targets_only(d_train)
    val_targets = data_utils.get_targets_only(d_val)

    with torch.no_grad():
        train_c = proj_layer(target_features.detach())
        val_c = proj_layer(val_target_features.detach())

        train_mean = torch.mean(train_c, dim=0, keepdim=True)
        train_std = torch.std(train_c, dim=0, keepdim=True)

        train_c -= train_mean
        train_c /= train_std

        train_y = torch.LongTensor(train_targets)

        val_c -= train_mean
        val_c /= train_std

        val_y = torch.LongTensor(val_targets)

    return {
        "classes": classes,
        "concepts": concepts,
        "train_c": train_c,
        "train_y": train_y,
        "val_c": val_c,
        "val_y": val_y,
        "W_c": W_c.detach().cpu(),
        "proj_mean": train_mean.cpu(),
        "proj_std": train_std.cpu(),
    }


SHARED_FILES = ("train_c", "train_y", "val_c", "val_y", "W_c", "proj_mean", "proj_std")


def save_shared_concept_data(shared, out_dir, args):
    """Cache the shared stage so the other arm (baseline vs gated) can be trained on
    literally the same concept activations without recomputing the projection."""
    os.makedirs(out_dir, exist_ok=True)
    for k in SHARED_FILES:
        torch.save(shared[k], os.path.join(out_dir, f"{k}.pt"))
    with open(os.path.join(out_dir, "concepts.txt"), "w") as f:
        f.write("\n".join(shared["concepts"]))
    with open(os.path.join(out_dir, "classes.txt"), "w") as f:
        f.write("\n".join(shared["classes"]))
    with open(os.path.join(out_dir, "shared_args.json"), "w") as f:
        json.dump({k: v for k, v in vars(args).items() if k in (
            "dataset", "concept_set", "backbone", "clip_name", "feature_layer", "proj_steps", "proj_batch_size",
            "clip_cutoff", "interpretability_cutoff", "no_filter", "seed")}, f, indent=2)
    return out_dir


def load_shared_concept_data(load_dir, args):
    shared = {k: torch.load(os.path.join(load_dir, f"{k}.pt"), map_location="cpu") for k in SHARED_FILES}
    with open(os.path.join(load_dir, "concepts.txt")) as f:
        shared["concepts"] = [ln for ln in f.read().split("\n") if ln]
    with open(os.path.join(load_dir, "classes.txt")) as f:
        shared["classes"] = f.read().split("\n")
    meta_path = os.path.join(load_dir, "shared_args.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        if meta.get("dataset") != args.dataset:
            raise ValueError(f"{load_dir} was built for {meta.get('dataset')}, not {args.dataset}")
    print(f"Loaded shared concept data from {load_dir}")
    return shared


def train_baseline_probe(shared, args, seed):
    torch.manual_seed(seed)
    indexed_train_ds = IndexedTensorDataset(shared["train_c"], shared["train_y"])
    indexed_train_loader = DataLoader(indexed_train_ds, batch_size=args.saga_batch_size, shuffle=True)
    val_ds = TensorDataset(shared["val_c"], shared["val_y"])
    val_loader = DataLoader(val_ds, batch_size=args.saga_batch_size, shuffle=False)

    linear = torch.nn.Linear(shared["train_c"].shape[1], len(shared["classes"])).to(args.device)
    linear.weight.data.zero_()
    linear.bias.data.zero_()
    metadata = {'max_reg': {'nongrouped': args.lam}}
    output_proj = glm_saga(linear, indexed_train_loader, 0.1, args.n_iters, 0.99, epsilon=1, k=1,
                    val_loader=val_loader, do_zero=False, metadata=metadata,
                    n_ex=shared["train_c"].shape[0], n_classes=len(shared["classes"]))
    W_g = output_proj['path'][0]['weight']
    b_g = output_proj['path'][0]['bias']
    return W_g, b_g, None


def train_gated_probe(shared, args, seed):
    torch.manual_seed(seed)
    indexed_train_ds = IndexedTensorDataset(shared["train_c"], shared["train_y"])
    indexed_train_loader = DataLoader(indexed_train_ds, batch_size=args.saga_batch_size, shuffle=True)

    gate_probe = GatedProbe(shared["train_c"].shape[1], len(shared["classes"]),
                            gate_temperature=args.gate_temperature, gate_forward=args.gate_forward).to(args.device)
    optimizer = torch.optim.Adam(gate_probe.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(args.gate_epochs):
        gate_probe.gate_temperature = gate_temperature_schedule(
            epoch, args.gate_epochs, args.gate_temperature, args.gate_temperature_final)
        for X_batch, y_batch, _ in indexed_train_loader:
            X_batch, y_batch = X_batch.to(args.device), y_batch.to(args.device)
            optimizer.zero_grad()
            logits, gates_batch = gate_probe(X_batch)
            loss = (loss_fn(logits, y_batch)
                    + args.gate_lam * gates_batch.sum())
            loss.backward()
            optimizer.step()

    gate_logits = gate_probe.gate_logits.data.cpu()
    gates = gate_probe.gate_probs().detach().cpu()
    # what the forward pass applied: soft sigmoid, or the hard mask for gate_forward="hard"
    W_g = gate_probe.linear.weight.data.cpu() * gate_probe.gate_mask().detach().cpu().unsqueeze(0)
    b_g = gate_probe.linear.bias.data.cpu()
    return W_g, b_g, gate_logits, gates


def refit_gated_probe(shared, args, seed, W_g, b_g, gates):
    """Mask-and-refit: K = gates > tau, head refit on train_c[:, K]. W_g already has the gate
    folded in, so fold_gate=False. Returns (W_refit, b_refit, keep, hard-cut acc without refit)."""
    masked_acc = masked_accuracy(W_g, b_g, gates, shared["val_c"], shared["val_y"],
                                 tau=args.refit_tau, fold_gate=False, device=args.device)
    W_r, b_r, keep = prune_and_refit(
        W_g, b_g, gates, shared["train_c"], shared["train_y"], tau=args.refit_tau, fold_gate=False,
        epochs=args.refit_epochs, lr=1e-3, batch_size=args.saga_batch_size, device=args.device, seed=seed)
    return W_r, b_r, keep, masked_acc


def evaluate_accuracy(W_g, b_g, val_c, val_y, device):
    with torch.no_grad():
        logits = val_c.to(device) @ W_g.to(device).T + b_g.to(device)
        acc = (logits.argmax(dim=1).cpu() == val_y).float().mean().item()
    return acc


def count_open_concepts(W_g, gate_logits, tol=1e-5):
    if gate_logits is not None:
        return (torch.sigmoid(gate_logits) >= 0.5).sum().item()
    return (W_g.abs() > tol).any(dim=0).sum().item()


def summarize(name, accs, n_opens, n_concepts_total):
    print("[{}] acc: {:.4f} +/- {:.4f}  open concepts: {:.1f} +/- {:.1f} / {}".format(
        name, statistics.fmean(accs), statistics.pstdev(accs) if len(accs) > 1 else 0.0,
        statistics.fmean(n_opens), statistics.pstdev(n_opens) if len(n_opens) > 1 else 0.0,
        n_concepts_total))
    return {
        "acc_per_seed": accs,
        "acc_mean": statistics.fmean(accs),
        "acc_std": statistics.pstdev(accs) if len(accs) > 1 else 0.0,
        "n_open_per_seed": n_opens,
        "n_open_mean": statistics.fmean(n_opens),
        "n_open_std": statistics.pstdev(n_opens) if len(n_opens) > 1 else 0.0,
    }


def save_probe(session_dir, shared, args, seed, W_g, b_g, gate_logits, refit=None):
    """One sub-dir per seed, shaped like a train_cbm.py run dir (args.txt, W_c/proj_mean/proj_std
    linked from the session dir, W_g/b_g/gate_logits, and the mask-and-refit head if any)."""
    run_dir = os.path.join(session_dir, f"seed{seed}")
    os.makedirs(run_dir, exist_ok=True)
    torch.save(W_g, os.path.join(run_dir, "W_g.pt"))
    torch.save(b_g, os.path.join(run_dir, "b_g.pt"))
    if gate_logits is not None:
        torch.save(gate_logits, os.path.join(run_dir, "gate_logits.pt"))
    if refit is not None:
        W_r, b_r, keep = refit
        torch.save(W_r, os.path.join(run_dir, "W_refit.pt"))
        torch.save(b_r, os.path.join(run_dir, "b_refit.pt"))
        torch.save(keep, os.path.join(run_dir, "keep_refit.pt"))
    with open(os.path.join(run_dir, "args.txt"), "w") as f:
        json.dump({**vars(args), "seed": seed, "arm": "C" if args.gated else "A"}, f, indent=2)
    for name in ("W_c.pt", "proj_mean.pt", "proj_std.pt", "concepts.txt"):
        link = os.path.join(run_dir, name)
        if not os.path.exists(link):
            os.symlink(os.path.join("..", name), link)
    return run_dir


def init_session_dir(session_dir, shared):
    """Seed-independent files once at the top of the session dir; seed dirs symlink to them."""
    os.makedirs(session_dir, exist_ok=True)
    torch.save(shared["W_c"], os.path.join(session_dir, "W_c.pt"))
    torch.save(shared["proj_mean"], os.path.join(session_dir, "proj_mean.pt"))
    torch.save(shared["proj_std"], os.path.join(session_dir, "proj_std.pt"))
    with open(os.path.join(session_dir, "concepts.txt"), "w") as f:
        f.write("\n".join(shared["concepts"]))


def main(args):
    if args.concept_set is None:
        args.concept_set = "data/concept_sets/{}_filtered.txt".format(args.dataset)
    os.makedirs(args.results_dir, exist_ok=True)

    if args.load_shared:
        shared = load_shared_concept_data(args.load_shared, args)
    else:
        shared = build_shared_concept_data(args)
        if args.save_shared is not None:
            out = args.save_shared or os.path.join(
                args.results_dir, "shared_{}_{}".format(args.dataset, datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")))
            print("Saved shared concept data to {}".format(save_shared_concept_data(shared, out, args)))
    n_concepts_total = shared["train_c"].shape[1]
    print("Shared concept set: {} concepts, {} classes".format(n_concepts_total, len(shared["classes"])))

    label = "gated" if args.gated else "baseline"
    stamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
    session_dir = os.path.join(args.results_dir, "{}_{}_seed_sweep_{}".format(args.dataset, label, stamp))
    if args.save_probes:
        init_session_dir(session_dir, shared)
        print("Saving per-seed probes under {}".format(session_dir))
    results = {"baseline": {"acc": [], "n_open": []}, "gated": {"acc": [], "n_open": []},
               "gated_refit": {"acc": [], "n_open": [], "masked_acc": []}}

    for seed in args.seeds:
        print("=== seed {} ===".format(seed))

        if not args.gated:
            W_base, b_base, _ = train_baseline_probe(shared, args, seed)
            acc_base = evaluate_accuracy(W_base, b_base, shared["val_c"], shared["val_y"], args.device)
            n_open_base = count_open_concepts(W_base, None)
            print("[baseline] seed={} test_acc={:.4f} open_concepts={}/{}".format(
                seed, acc_base, n_open_base, n_concepts_total))
            results["baseline"]["acc"].append(acc_base)
            results["baseline"]["n_open"].append(n_open_base)
            if args.save_probes:
                save_probe(session_dir, shared, args, seed, W_base, b_base, None)
            continue

        W_gate, b_gate, gate_logits, gates = train_gated_probe(shared, args, seed)
        acc_gate = evaluate_accuracy(W_gate, b_gate, shared["val_c"], shared["val_y"], args.device)
        n_open_gate = count_open_concepts(W_gate, gate_logits)
        print("[gated]    seed={} test_acc={:.4f} open_concepts={}/{}".format(
            seed, acc_gate, n_open_gate, n_concepts_total))
        results["gated"]["acc"].append(acc_gate)
        results["gated"]["n_open"].append(n_open_gate)

        refit = None
        if not args.no_refit:
            W_r, b_r, keep, masked_acc = refit_gated_probe(shared, args, seed, W_gate, b_gate, gates)
            acc_refit = evaluate_head(W_r, b_r, shared["val_c"], shared["val_y"], device=args.device)
            print("[gated_refit] seed={} concepts={}/{} @tau={}  hard cut {:.4f} -> refit {:.4f}".format(
                seed, int(keep.sum()), n_concepts_total, args.refit_tau, masked_acc, acc_refit))
            results["gated_refit"]["acc"].append(acc_refit)
            results["gated_refit"]["n_open"].append(int(keep.sum()))
            results["gated_refit"]["masked_acc"].append(masked_acc)
            refit = (W_r, b_r, keep)
        if args.save_probes:
            save_probe(session_dir, shared, args, seed, W_gate, b_gate, gate_logits, refit)

    summary = {
        "dataset": args.dataset,
        "concept_set": args.concept_set,
        "shared_from": args.load_shared,
        "seeds": args.seeds,
        "n_concepts_total": n_concepts_total,
        "lam": args.lam,
        "gate_lam": args.gate_lam,
        "gate_epochs": args.gate_epochs,
        "gate_forward": args.gate_forward,
    }
    summary["arm"] = label
    if results["baseline"]["acc"]:
        summary["baseline"] = summarize("baseline", results["baseline"]["acc"], results["baseline"]["n_open"], n_concepts_total)
    if results["gated"]["acc"]:
        summary["gated"] = summarize("gated", results["gated"]["acc"], results["gated"]["n_open"], n_concepts_total)
    if results["gated_refit"]["acc"]:
        summary["refit_tau"] = args.refit_tau
        summary["gated_refit"] = summarize("gated_refit", results["gated_refit"]["acc"],
                                           results["gated_refit"]["n_open"], n_concepts_total)
        summary["gated_refit"]["masked_acc_per_seed"] = results["gated_refit"]["masked_acc"]
        summary["gated_refit"]["masked_acc_mean"] = statistics.fmean(results["gated_refit"]["masked_acc"])

    summary["session_dir"] = session_dir if args.save_probes else None
    out_path = os.path.join(args.results_dir, "{}_{}_seed_sweep_{}.json".format(args.dataset, label, stamp))
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    if args.save_probes:  # a copy next to the probes it describes
        with open(os.path.join(session_dir, "seed_sweep_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
    print("Saved results to {}".format(out_path))


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
