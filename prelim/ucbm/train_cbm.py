import torch
from os import path, makedirs
import json
import random

from data_loader import load_data
import numpy as np
from model_loader import get_model
from plotter.plotter import Plotter
from ucbm import UCBM
import argparse
from datetime import datetime
from constants import DATA_SETS, MODELS, RESULT_PATH

def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Load plotter object.
    plotter = Plotter(path.join(RESULT_PATH, f'{args.dataset}-{args.backbone}/'))
    print("Plotter loaded successfully...")

    # Read concept bank.
    act_bank_path = con_bank_path = path.join(plotter.get_concept_data_path(), args.concept_data)

    # Load models.
    model = get_model(args.backbone, args.device)

    print(f"Model {args.backbone} loaded successfully...")

    training_data = load_data(args.dataset, True, 
                                model.transform, reorder_idcs=True)
    test_data = load_data(args.dataset, False, 
                            model.transform, reorder_idcs=True)
    print(f"Dataset {args.dataset} loaded successfully...")

    if not path.exists(con_bank_path):
        raise AttributeError(f"Concept bank {args.concept_data} does not exist")
    h = np.load(path.join(con_bank_path, "h.npy"))


    # print(f"Read concept bank {con_bank_path} successfully...")

    # Train the post-hoc cbms. 
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
        gated=args.gated,
        gate_temperature=args.gate_temperature,
        gate_temperature_final=args.gate_temperature_final,
        gate_forward=args.gate_forward)
    # ph_cbm.fit(training_data, con_bank_path, test_data)
    ph_cbm.fit(training_data, act_bank_path, None, cocostuff_training=(args.dataset=="cocostuff"))    
    print("Trained classifier...")
    
    label = "gated" if args.gated else "baseline"
    save_name = args.cls_save_name
    if save_name == "":
        save_name = f"{label}_{datetime.now().strftime('%Y_%m_%d_-_%H_%M_%S')}"
    else:
        save_name += f"-{label}-{datetime.now().strftime('%Y_%m_%d_-_%H_%M_%S')}"
    
    class_path = path.join(plotter.get_classifier_path(), args.concept_data, save_name)
    makedirs(class_path)
    ph_cbm.save_to_file(class_path, "classifier.pth")
    
    if args.dataset in ["imagenet", "places365", "imagenet100", "cub"]:
        metrics = ["acc"]
    else:
        metrics = ["acc", "auprc", "auprc_pc", "auroc"]
    info_dict = ph_cbm.get_info_dict(training_data, test_data, act_bank_path, metrics=metrics)
    if args.gated and not args.no_refit:
        # Mask-and-refit: the open set at tau as a hard mask, head refit on it. This is the
        # model "amount of concepts" describes; its accuracy goes in info.json under "refit".
        refit = ph_cbm.prune_and_refit(training_data, test_data, act_bank_path, tau=args.refit_tau,
                                       epochs=args.refit_epochs)
        torch.save({"weight": refit["W_refit"], "bias": refit["b_refit"], "keep": refit["keep"],
                    "tau": args.refit_tau}, path.join(class_path, "classifier_refit.pt"))
        info_dict["refit"] = {k: v for k, v in refit.items() if not torch.is_tensor(v)}
        print(f"Mask-and-refit @tau={args.refit_tau}: {refit['n_concepts']} concepts, "
              f"hard cut {refit['masked_accuracy_no_refit']:.4f} -> refit {refit['test_accuracy']:.4f}")
    print(json.dumps(info_dict, indent=2))
    with open(path.join(class_path, "info.json"), "w") as f:
        json.dump(info_dict, f, indent=2)
    print(f"Saved information to {class_path}")

    if args.gated:
        plotter.plot_gate_distribution(ph_cbm, tmp=save_name, save_dir=class_path)
        print("Saved gate distribution plot...")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Settings for creating PCBM-U')

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
                        help="Batch size used to1 train the linear model")
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
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("-g", "--gated", action="store_true")
    parser.add_argument("--gate_forward", choices=["soft", "hard"], default="soft",
                        help="gated: 'soft' multiplies concepts by sigmoid(gate); 'hard' uses the mask 1[sigmoid>0.5] in "
                             "the forward pass with a straight-through gradient, so the trained model is exactly the "
                             "masked one and gates cannot drift below 0.5 for free")
    parser.add_argument("--refit_tau", type=float, default=0.5,
                        help="gated: keep concepts with gate > tau, drop the rest, refit the head on the survivors "
                             "(saved as classifier_refit.pt; accuracy in info.json under 'refit')")
    parser.add_argument("--refit_epochs", type=int, default=20)
    parser.add_argument("--no_refit", action="store_true", help="skip the mask-and-refit stage")
    parser.add_argument("--gate_temperature", type=float, default=1.0,
                        help="Temperature T dividing gate_logits before the sigmoid (only used with --gated). "
                             "T < 1 sharpens the gate toward 0/1 (more bimodal); T = 1 is the plain sigmoid. "
                             "Starting temperature of the anneal if --gate_temperature_final is also given.")
    parser.add_argument("--gate_temperature_final", type=float, default=None,
                        help="If given (with --gated), anneal the gate temperature geometrically from "
                             "--gate_temperature down (or up) to this value over training, instead of keeping "
                             "it fixed. E.g. --gate_temperature 1.0 --gate_temperature_final 0.2")

    args = parser.parse_args()
    main(args)