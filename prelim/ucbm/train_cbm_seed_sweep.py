import torch
from os import path, makedirs
import json
import random
import statistics

from data_loader import load_data
import numpy as np
from model_loader import get_model
from plotter.plotter import Plotter
from ucbm import UCBM
import argparse
from datetime import datetime
from constants import DATA_SETS, MODELS, RESULT_PATH

"""
Trains either the baseline (Classifier) or gated (GatedProbe) probe -- pass
-g/--gated to train gated, omit it to train baseline -- the same way
train_cbm.py does, but repeats probe training across multiple seeds. The
backbone/concept-bank activations are seed-independent and get cached to
disk by UCBM._get_concept_embeddings on first use, so only the probe itself
is retrained per seed. Records test accuracy and number of open concepts for
each seed.
"""


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_open_concepts(ph_cbm, tol=1e-5):
    """Number of concepts that can ever affect a prediction. For gated
    models this is the learned gate; a baseline Classifier has no gate, so a
    concept counts as open if any class has a nonzero weight on it -- the
    same criterion used for the Label-free-CBM baseline sweep, so gated and
    baseline are measured on the same footing."""
    if ph_cbm._gated:
        return int((ph_cbm._classifier.gate_probs() > 0.5).sum().item())
    weight = ph_cbm._classifier.linear.weight.data
    return int((weight.abs() > tol).any(dim=0).sum().item())


def train_one(args, seed, gated, model, training_data, test_data, act_bank_path, h, session_dir=None):
    set_seed(seed)

    ph_cbm = UCBM(
        backbone=model.g,
        h=h,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lam_pi=args.lam_pi,
        lambda_gate=args.lambda_gate,
        lam_w=args.lam_w,
        dropout_p=args.dropout_p,
        learning_rate=args.lr,
        relu=args.relu,
        scale_mode=args.scale_choose,
        bias_mode=args.bias_choose,
        normalize=args.normalize_concepts,
        k=args.k,
        device=args.device,
        gated=gated,
        gate_temperature=args.gate_temperature,
        gate_temperature_final=args.gate_temperature_final,
        gate_forward=args.gate_forward)
    ph_cbm.fit(training_data, act_bank_path, None, cocostuff_training=(args.dataset == "cocostuff"))

    if args.dataset in ["imagenet", "places365", "imagenet100", "cub"]:
        metrics = ["acc"]
    else:
        metrics = ["acc", "auprc", "auprc_pc", "auroc"]
    info_dict = ph_cbm.get_info_dict(training_data, test_data, act_bank_path, metrics=metrics)

    test_acc = info_dict["test acc"]
    n_open = count_open_concepts(ph_cbm)

    refit = None
    if gated and not args.no_refit:
        refit = ph_cbm.prune_and_refit(training_data, test_data, act_bank_path, tau=args.refit_tau,
                                       epochs=args.refit_epochs, seed=seed)
        print(f"[gated_refit] seed={seed} concepts={refit['n_concepts']}/{h.shape[0]} @tau={args.refit_tau}  "
              f"hard cut {refit['masked_accuracy_no_refit']:.4f} -> refit {refit['test_accuracy']:.4f}")
        info_dict["refit"] = {k: v for k, v in refit.items() if not torch.is_tensor(v)}

    if args.save_classifiers:
        # one folder per sweep session, one sub-dir per seed (concept_retention.py --run-dir
        # accepts the session dir and picks up every seed*/classifier.pth under it)
        class_path = path.join(session_dir, f"seed{seed}")
        makedirs(class_path, exist_ok=True)
        ph_cbm.save_to_file(class_path, "classifier.pth")
        with open(path.join(class_path, "info.json"), "w") as f:
            json.dump(info_dict, f, indent=2)
        if gated:
            plotter.plot_gate_distribution(ph_cbm, tmp=f"seed{seed}", save_dir=class_path)
        if refit is not None:
            torch.save({"weight": refit["W_refit"], "bias": refit["b_refit"], "keep": refit["keep"],
                        "tau": args.refit_tau}, path.join(class_path, "classifier_refit.pt"))

    return test_acc, n_open, refit


def summarize(name, accs, n_opens):
    print("[{}] test_acc: {:.4f} +/- {:.4f}  open concepts: {:.1f} +/- {:.1f}".format(
        name, statistics.fmean(accs), statistics.pstdev(accs) if len(accs) > 1 else 0.0,
        statistics.fmean(n_opens), statistics.pstdev(n_opens) if len(n_opens) > 1 else 0.0))
    return {
        "acc_per_seed": accs,
        "acc_mean": statistics.fmean(accs),
        "acc_std": statistics.pstdev(accs) if len(accs) > 1 else 0.0,
        "n_open_per_seed": n_opens,
        "n_open_mean": statistics.fmean(n_opens),
        "n_open_std": statistics.pstdev(n_opens) if len(n_opens) > 1 else 0.0,
    }


def main(args):
    global plotter
    plotter = Plotter(path.join(RESULT_PATH, f'{args.dataset}-{args.backbone}/'))
    print("Plotter loaded successfully...")

    act_bank_path = con_bank_path = path.join(plotter.get_concept_data_path(), args.concept_data)

    model = get_model(args.backbone, args.device)
    print(f"Model {args.backbone} loaded successfully...")

    training_data = load_data(args.dataset, True, model.transform, reorder_idcs=True)
    test_data = load_data(args.dataset, False, model.transform, reorder_idcs=True)
    print(f"Dataset {args.dataset} loaded successfully...")

    if not path.exists(con_bank_path):
        raise AttributeError(f"Concept bank {args.concept_data} does not exist")
    h = np.load(path.join(con_bank_path, "h.npy"))

    label = "gated" if args.gated else "baseline"
    stamp = datetime.now().strftime('%Y_%m_%d_%H_%M')
    session_dir = path.join(plotter.get_classifier_path(), args.concept_data,
                            f"{args.cls_save_name}{'-' if args.cls_save_name else ''}{label}_seed_sweep_{stamp}")
    if args.save_classifiers:
        makedirs(session_dir, exist_ok=True)
        print(f"Saving per-seed classifiers under {session_dir}")
    accs, n_opens = [], []
    refit_accs, refit_n, masked_accs = [], [], []

    for seed in args.seeds:
        print(f"=== seed {seed} ===")

        acc, n_open, refit = train_one(args, seed, args.gated, model, training_data, test_data, act_bank_path, h,
                                       session_dir=session_dir)
        print(f"[{label}] seed={seed} test_acc={acc:.4f} open_concepts={n_open}/{h.shape[0]}")

        accs.append(acc)
        n_opens.append(n_open)
        if refit is not None:
            refit_accs.append(refit["test_accuracy"])
            refit_n.append(refit["n_concepts"])
            masked_accs.append(refit["masked_accuracy_no_refit"])

    summary = {
        "dataset": args.dataset,
        "backbone": args.backbone,
        "concept_data": args.concept_data,
        "gated": args.gated,
        "seeds": args.seeds,
        "n_concepts_total": int(h.shape[0]),
        "lam_pi": args.lam_pi,
        "lambda_gate": args.lambda_gate,
        "gate_forward": args.gate_forward,
        "lam_w": args.lam_w,
        "epochs": args.epochs,
    }
    summary[label] = summarize(label, accs, n_opens)
    if refit_accs:
        summary["refit_tau"] = args.refit_tau
        summary["gated_refit"] = summarize("gated_refit", refit_accs, refit_n)
        summary["gated_refit"]["masked_acc_per_seed"] = masked_accs
        summary["gated_refit"]["masked_acc_mean"] = statistics.fmean(masked_accs)

    results_dir = path.join(plotter.get_classifier_path(), args.concept_data, "seed_sweep_results")
    makedirs(results_dir, exist_ok=True)
    out_path = path.join(results_dir, f"{args.dataset}_{label}_seed_sweep_{stamp}.json")
    summary["session_dir"] = session_dir if args.save_classifiers else None
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    if args.save_classifiers:  # a copy next to the probes it describes
        with open(path.join(session_dir, "seed_sweep_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
    print(f"Saved results to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train baseline and gated PCBM-U probes across multiple seeds')

    parser.add_argument("-d", "--dataset", type=str, default="imagenet",
                        help="The dataset to load",
                        choices=DATA_SETS)
    parser.add_argument("-b", "--backbone", type=str, default="resnet50_v2",
                        help="Which pretrained model backbone to use",
                        choices=MODELS)
    parser.add_argument("-c", "--concept_data", type=str, default="",
                        help="Name of the concept data to use")

    parser.add_argument("--device", type=str,
                        default=("cuda" if torch.cuda.is_available() else "cpu"),
                        help="Which device to use", choices=["cpu", "cuda"])
    parser.add_argument("--relu", type=str, default="ReLU",
                        help="relu function to use",
                        choices=["no", "ReLU", "jumpReLU"])

    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size used to train the linear model")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of epochs to train the post-hoc cbm")
    parser.add_argument("--scale_choose", type=str, default="learn",
                        choices=["learn", "no"],
                        help="How scale is chosen")
    parser.add_argument("--bias_choose", type=str, default="learn",
                        choices=["learn", "no"],
                        help="How scale is chosen")
    parser.add_argument("--normalize_concepts", action="store_true")
    parser.add_argument("--k", type=int, default=-1,
                        help="Top K concepts to keep")
    parser.add_argument("--lam_pi", type=float, default=1e-4,
                        help="Elastic-net regularization strength on the concept selector's output pi(x), independent of gating")
    parser.add_argument("--lambda_gate", type=float, default=1e-4,
                        help="Regularization strength for the learned per-concept gate's sparsity penalty (only used with --gated)")
    parser.add_argument("--lam_w", type=float, default=1e-4,
                        help="Factor for weight regularization of final layer")
    parser.add_argument("--dropout_p", type=float, default=0.2,
                        help="Dropout rate for dropping for concept dropping")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate to train the model with")
    parser.add_argument("--cls_save_name", type=str, default="")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 10, 20, 30, 40],
                        help="Seeds to train each probe type with; results are reported per-seed and averaged")
    parser.add_argument("--save_classifiers", action="store_true",
                        help="Also save each seed's trained classifier + info.json, like trainss_cbm.py does for a single run")
    parser.add_argument("--gate_forward", choices=["soft", "hard"], default="soft",
                        help="gated: 'soft' multiplies concepts by sigmoid(gate); 'hard' uses the mask 1[sigmoid>0.5] in "
                             "the forward pass with a straight-through gradient, so the trained model is exactly the "
                             "masked one and gates cannot drift below 0.5 for free")
    parser.add_argument("--refit_tau", type=float, default=0.5,
                        help="gated: keep concepts with gate > tau, drop the rest, refit the head on the survivors "
                             "(reported as 'gated_refit'; saved as classifier_refit.pt with --save_classifiers)")
    parser.add_argument("--refit_epochs", type=int, default=20)
    parser.add_argument("--no_refit", action="store_true", help="skip the mask-and-refit stage")
    parser.add_argument("-g", "--gated", action="store_true",
                        help="Train the gated probe across seeds; omit to train the baseline probe instead")
    parser.add_argument("--gate_temperature", type=float, default=1.0,
                        help="Temperature T dividing gate_logits before the sigmoid (only used with --gated). "
                             "T < 1 sharpens the gate toward 0/1 (more bimodal); T = 1 is the plain sigmoid. "
                             "Starting temperature of the anneal if --gate_temperature_final is also given.")
    parser.add_argument("--gate_temperature_final", type=float, default=None,
                        help="If given (with --gated), anneal the gate temperature geometrically from "
                             "--gate_temperature down (or up) to this value over training, instead of keeping "
                             "it fixed.")

    args = parser.parse_args()
    main(args)
