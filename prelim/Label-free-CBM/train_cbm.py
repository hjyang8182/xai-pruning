import sys
import torch
import os
import random
import utils
import data_utils
import similarity
import argparse
import datetime
import json
import matplotlib.pyplot as plt

from glm_saga.elasticnet import IndexedTensorDataset, glm_saga
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from gated_probe import GatedProbe, gate_temperature_schedule, prune_and_refit, masked_accuracy, evaluate_head
from concept_usage import concept_usage_report  # shared backbone-agnostic metric at repo root

parser = argparse.ArgumentParser(description='Settings for creating CBM')


parser.add_argument("-d", "--dataset", type=str, default="cifar10")
parser.add_argument("--concept_set", type=str, default=None, 
                    help="path to concept set name")
parser.add_argument("--backbone", type=str, default="clip_RN50", help="Which pretrained model to use as backbone")
parser.add_argument("--clip_name", type=str, default="ViT-B/16", help="Which CLIP model to use")

parser.add_argument("--device", type=str, default="cuda", help="Which device to use")
parser.add_argument("--batch_size", type=int, default=512, help="Batch size used when saving model/CLIP activations")
parser.add_argument("--saga_batch_size", type=int, default=256, help="Batch size used when fitting final layer")
parser.add_argument("--proj_batch_size", type=int, default=50000, help="Batch size to use when learning projection layer")

parser.add_argument("--feature_layer", type=str, default='layer4', 
                    help="Which layer to collect activations from. Should be the name of second to last layer in the model")
parser.add_argument("--activation_dir", type=str, default='saved_activations', help="save location for backbone and CLIP activations")
parser.add_argument("--save_dir", type=str, default='saved_models', help="where to save trained models")
parser.add_argument("--clip_cutoff", type=float, default=0.25, help="concepts with smaller top5 clip activation will be deleted")
parser.add_argument("--proj_steps", type=int, default=1000, help="how many steps to train the projection layer for")
parser.add_argument("--proj_patience", type=int, default=1, help="number of consecutive non-improving val checks before stopping projection training (each check is 50 steps)")
parser.add_argument("--interpretability_cutoff", type=float, default=0.45, help="concepts with smaller similarity to target concept will be deleted")
parser.add_argument("--lam", type=float, default=0.0007, help="Sparsity regularization parameter, higher->more sparse")
parser.add_argument("--n_iters", type=int, default=1000, help="How many iterations to run the final layer solver for")
parser.add_argument("--gate_epochs", type=int, default=20, help="How many epochs to train the arm-B/arm-C probe for")
parser.add_argument("--lambda_gate", type=float, default=1e-4, help="Arm C only: sparsity penalty lambda_gate * sum(sigmoid(gate_logits)) on the learned gate")
parser.add_argument("--gate_temperature", type=float, default=1.0, help="Arm C only: temperature T dividing gate_logits before the sigmoid. T < 1 sharpens the gate toward 0/1 (more bimodal); T = 1 is the plain sigmoid. Starting temperature of the anneal if --gate_temperature_final is also given.")
parser.add_argument("--gate_forward", choices=["soft", "hard"], default="soft", help="Arm C only: 'soft' multiplies concepts by sigmoid(gate); 'hard' uses the mask 1[sigmoid>0.5] in the forward pass with a straight-through gradient, so the trained model is exactly the masked model and gates cannot drift below 0.5 for free.")
parser.add_argument("--refit_tau", type=float, default=0.5, help="Arm C only: after training, keep concepts with gate > tau, drop the rest, and refit the head on the survivors; saved as W_refit.pt/b_refit.pt with its own test accuracy in metrics.txt.")
parser.add_argument("--refit_epochs", type=int, default=20)
parser.add_argument("--no_refit", action="store_true", help="Arm C only: skip the mask-and-refit stage")
parser.add_argument("--gate_temperature_final", type=float, default=None, help="Arm C only: if given, anneal the gate temperature geometrically from --gate_temperature down (or up) to this value over training, instead of keeping it fixed.")
parser.add_argument("--print", action='store_true', help="Print all concepts being deleted in this stage")
parser.add_argument("--arm", type=str, default="A", choices=["A", "B", "C"],
                     help="A: GLM-SAGA final layer, as published (default). "
                          "B: same Adam/CE final-layer training as C, but no gate -- same-optimiser control. "
                          "C: Adam/CE final layer with a learned global gate (CE + lambda_gate * sum(sigmoid(gate_logits))).")
parser.add_argument("--no_filter", action="store_true", help="Skip CLIP and interpretability filtering; use full concept set (recommended with --arm C)")

def train_cbm_and_save(args):
    
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    if args.concept_set==None:
        args.concept_set = "data/concept_sets/{}_filtered.txt".format(args.dataset)
        
    similarity_fn = similarity.cos_similarity_cubed_single
    
    d_train = args.dataset + "_train"
    # CUB only has an official train/test split, no val -- everywhere else calls
    # this eval split "_val", but for cub it's really "_test".
    d_val = args.dataset + ("_test" if args.dataset == "cub" else "_val")

    #get concept set
    cls_file = data_utils.LABEL_FILES[args.dataset]
    with open(cls_file, "r") as f:
        classes = f.read().split("\n")
    
    with open(args.concept_set) as f:
        concepts = f.read().split("\n")
    
    #save activations and get save_paths
    for d_probe in [d_train, d_val]:
        utils.save_activations(clip_name = args.clip_name, target_name = args.backbone, 
                               target_layers = [args.feature_layer], d_probe = d_probe,
                               concept_set = args.concept_set, batch_size = args.batch_size, 
                               device = args.device, pool_mode = "avg", save_dir = args.activation_dir)
        
    target_save_name, clip_save_name, text_save_name = utils.get_save_names(args.clip_name, args.backbone, 
                                            args.feature_layer,d_train, args.concept_set, "avg", args.activation_dir)
    val_target_save_name, val_clip_save_name, text_save_name =  utils.get_save_names(args.clip_name, args.backbone,
                                            args.feature_layer, d_val, args.concept_set, "avg", args.activation_dir)
    
    #load features
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
    
    #filter concepts not activating highly
    highest = torch.mean(torch.topk(clip_features, dim=0, k=5)[0], dim=0)

    if not args.no_filter:
        if args.print:
            for i, concept in enumerate(concepts):
                if highest[i]<=args.clip_cutoff:
                    print("Deleting {}, CLIP top5:{:.3f}".format(concept, highest[i]))
        concepts = [concepts[i] for i in range(len(concepts)) if highest[i]>args.clip_cutoff]
        clip_mask = highest > args.clip_cutoff
    else:
        clip_mask = torch.ones(len(highest), dtype=torch.bool)

    #save memory by recalculating
    del clip_features
    with torch.no_grad():
        image_features = torch.load(clip_save_name, map_location="cpu").float()
        image_features /= torch.norm(image_features, dim=1, keepdim=True)

        text_features = torch.load(text_save_name, map_location="cpu").float()[clip_mask]
        text_features /= torch.norm(text_features, dim=1, keepdim=True)

        clip_features = image_features @ text_features.T
        del image_features, text_features

    val_clip_features = val_clip_features[:, clip_mask]
    
    #learn projection layer
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
        if i%50==0 or i==args.proj_steps-1:
            with torch.no_grad():
                val_output = proj_layer(val_target_features.to(args.device).detach())
                val_loss = -similarity_fn(val_clip_features.to(args.device).detach(), val_output)
                val_loss = torch.mean(val_loss)
            if i==0:
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
        
    proj_layer.load_state_dict({"weight":best_weights})
    print("Best step:{}, Avg val similarity:{:.4f}".format(best_step, -best_val_loss.cpu()))
    
    #delete concepts that are not interpretable
    with torch.no_grad():
        outs = proj_layer(val_target_features.to(args.device).detach())
        sim = similarity_fn(val_clip_features.to(args.device).detach(), outs)
        interpretable = sim > args.interpretability_cutoff

    if not args.no_filter:
        if args.print:
            for i, concept in enumerate(concepts):
                if sim[i]<=args.interpretability_cutoff:
                    print("Deleting {}, Iterpretability:{:.3f}".format(concept, sim[i]))
        concepts = [concepts[i] for i in range(len(concepts)) if interpretable[i]]
        W_c = proj_layer.weight[interpretable]
    else:
        W_c = proj_layer.weight

    del clip_features, val_clip_features
    proj_layer = torch.nn.Linear(in_features=target_features.shape[1], out_features=len(concepts), bias=False)
    proj_layer.load_state_dict({"weight":W_c})
    
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
        indexed_train_ds = IndexedTensorDataset(train_c, train_y)

        val_c -= train_mean
        val_c /= train_std
        
        val_y = torch.LongTensor(val_targets)

        val_ds = TensorDataset(val_c,val_y)


    indexed_train_loader = DataLoader(indexed_train_ds, batch_size=args.saga_batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.saga_batch_size, shuffle=False)

    STEP_SIZE = 0.1
    ALPHA = 0.99
    metadata = {'max_reg': {'nongrouped': args.lam}}

    if args.arm == "C":
        # Jointly learn a global gate and the final linear layer with Adam:
        # L = CE + lambda_gate * sum(sigmoid(gate_logits)), reduction='sum' on
        # the gate term (gates_batch is the raw (n_concepts,) sigmoid vector,
        # not per-example, so .sum() sums over concepts). No GLM-SAGA stage.
        gate_probe = GatedProbe(train_c.shape[1], len(classes), gate_temperature=args.gate_temperature,
                                gate_forward=args.gate_forward).to(args.device)
        optimizer = torch.optim.Adam(gate_probe.parameters(), lr=1e-3)
        loss_fn = torch.nn.CrossEntropyLoss()

        gate_log_every = 500
        n_gate_iters = args.gate_epochs * len(indexed_train_loader)
        step = 0
        for epoch in range(args.gate_epochs):
            gate_probe.gate_temperature = gate_temperature_schedule(
                epoch, args.gate_epochs, args.gate_temperature, args.gate_temperature_final)
            for X_batch, y_batch, _ in indexed_train_loader:
                X_batch, y_batch = X_batch.to(args.device), y_batch.to(args.device)
                optimizer.zero_grad()
                logits, gates_batch = gate_probe(X_batch)
                loss = (loss_fn(logits, y_batch)
                        + args.lambda_gate * gates_batch.sum())
                loss.backward()
                optimizer.step()
                step += 1

                if step % gate_log_every == 0 or step == n_gate_iters:
                    with torch.no_grad():
                        cur_gates = gate_probe.gate_probs()
                        pct_closed = (cur_gates < 0.5).float().mean().item()
                    print("gate epoch {}/{} step {}/{}: T={:.4f} mean_gate={:.4f} pct_closed={:.4f}".format(
                        epoch + 1, args.gate_epochs, step, n_gate_iters, gate_probe.gate_temperature,
                        cur_gates.mean().item(), pct_closed))

        # Fold the learned gates into the linear weight columns so the saved
        # W_g/b_g can be applied directly to ungated concept activations at
        # inference time (self.final(proj_c) in cbm.py), matching the
        # gate_probe's forward (linear(x * gates)).
        gate_logits = gate_probe.gate_logits.data.cpu()
        gates = gate_probe.gate_probs().detach().cpu()  # (n_concepts,) -- reflects the converged gate_temperature
        # W_g stores the weights after gating -- exactly what the forward pass applied: the soft
        # sigmoid for gate_forward="soft", the hard mask 1[g > 0.5] for "hard".
        W_g = (gate_probe.linear.weight.data.cpu() * gate_probe.gate_mask().detach().cpu().unsqueeze(0))
        b_g = gate_probe.linear.bias.data.cpu()
    elif args.arm == "B":
        # Same optimiser/loss/epoch budget as arm C minus the gate term --
        # isolates whatever arm C's gate contributes, since B and C differ by
        # exactly one term.
        gate_logits = None
        probe = torch.nn.Linear(train_c.shape[1], len(classes)).to(args.device)
        optimizer = torch.optim.Adam(probe.parameters(), lr=1e-3)
        loss_fn = torch.nn.CrossEntropyLoss()
        for epoch in range(args.gate_epochs):
            for X_batch, y_batch, _ in indexed_train_loader:
                X_batch, y_batch = X_batch.to(args.device), y_batch.to(args.device)
                optimizer.zero_grad()
                loss = loss_fn(probe(X_batch), y_batch)
                loss.backward()
                optimizer.step()
        W_g = probe.weight.data.cpu()
        b_g = probe.bias.data.cpu()
    else:
        gate_logits = None
        linear = torch.nn.Linear(train_c.shape[1], len(classes)).to(args.device)
        linear.weight.data.zero_()
        linear.bias.data.zero_()
        output_proj = glm_saga(linear, indexed_train_loader, STEP_SIZE, args.n_iters, ALPHA, epsilon=1, k=1,
                        val_loader=val_loader, do_zero=False, metadata=metadata, n_ex=len(target_features), n_classes=len(classes))
        W_g = output_proj['path'][0]['weight']
        b_g = output_proj['path'][0]['bias']

    with torch.no_grad():
        test_logits = val_c.to(args.device) @ W_g.to(args.device).T + b_g.to(args.device)
        test_preds = test_logits.argmax(dim=1).cpu()
        test_acc = (test_preds == val_y).float().mean().item()
    print("Test accuracy:{:.4f}".format(test_acc))

    arm_label = {"A": "armA_glmsaga", "B": "armB_ce", "C": "armC_gated"}[args.arm]
    save_name = "{}/{}_cbm_{}_{}".format(args.save_dir, args.dataset, arm_label, datetime.datetime.now().strftime("%Y_%m_%d_%H_%M"))
    os.mkdir(save_name)
    torch.save(train_mean, os.path.join(save_name, "proj_mean.pt"))
    torch.save(train_std, os.path.join(save_name, "proj_std.pt"))
    torch.save(W_c, os.path.join(save_name ,"W_c.pt"))
    torch.save(W_g, os.path.join(save_name, "W_g.pt"))
    torch.save(b_g, os.path.join(save_name, "b_g.pt"))
    refit = None
    if gate_logits is not None:
        torch.save(gate_logits, os.path.join(save_name, "gate_logits.pt"))
        if not args.no_refit:
            # The deliverable of the gated run: the open set K = gates > tau as a hard mask,
            # head refit on train_c[:, K]. W_g has the gate folded in -> fold_gate=False.
            masked_acc = masked_accuracy(W_g, b_g, gates, val_c, val_y, tau=args.refit_tau,
                                         fold_gate=False, device=args.device)
            W_r, b_r, keep = prune_and_refit(
                W_g, b_g, gates, train_c, train_y, tau=args.refit_tau, fold_gate=False,
                epochs=args.refit_epochs, lr=1e-3, batch_size=args.saga_batch_size, device=args.device)
            refit_acc = evaluate_head(W_r, b_r, val_c, val_y, device=args.device)
            torch.save(W_r, os.path.join(save_name, "W_refit.pt"))
            torch.save(b_r, os.path.join(save_name, "b_refit.pt"))
            refit = {"tau": args.refit_tau, "n_concepts": int(keep.sum()),
                     "masked_accuracy_no_refit": masked_acc, "test_accuracy": refit_acc}
            print("Mask-and-refit @tau={}: {} concepts, hard cut {:.4f} -> refit {:.4f}".format(
                args.refit_tau, int(keep.sum()), masked_acc, refit_acc))
        plt.figure(figsize=(8, 4))
        plt.hist(gates.numpy(), bins=50)
        plt.axvline(x=0.5, color='red', linestyle='--', label='threshold')
        plt.xlabel('Gate value')
        plt.ylabel('Count')
        plt.title('Distribution of gate values')
        plt.legend()
        plt.savefig(os.path.join(save_name, "gate_distribution.png"), bbox_inches='tight', dpi=150)
        plt.close()

    with open(os.path.join(save_name, "concepts.txt"), 'w') as f:
        f.write(concepts[0])
        for concept in concepts[1:]:
            f.write('\n'+concept)
    
    with open(os.path.join(save_name, "args.txt"), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    
    with open(os.path.join(save_name, "metrics.txt"), 'w') as f:
        out_dict = {'arm': args.arm}
        if args.arm == "A":
            for key in ('lam', 'lr', 'alpha', 'time'):
                out_dict[key] = float(output_proj['path'][0][key])
            out_dict['metrics'] = output_proj['path'][0]['metrics']
        elif args.arm == "C":
            out_dict['lambda_gate'] = args.lambda_gate
            out_dict['gate_forward'] = args.gate_forward
            out_dict['gate_temperature_start'] = args.gate_temperature
            out_dict['gate_temperature_final'] = args.gate_temperature_final
            out_dict['gate_temperature_used'] = gate_probe.gate_temperature
            # "open" is sigmoid(gate/T) >= 0.5, i.e. gate_logits >= 0 -- the
            # 0.5 threshold on the raw logit is unaffected by temperature.
            n_open = (gate_logits >= 0).sum().item()
            n_gates = gate_logits.numel()
            out_dict['gates'] = {"mean_gate": gates.mean().item(),
                                  "Open gates": n_open, "Total gates": n_gates,
                                  "Percentage open": n_open / n_gates}
        nnz = (W_g.abs() > 1e-5).sum().item()
        total = W_g.numel()
        out_dict['sparsity'] = {"Non-zero weights":nnz, "Total weights":total, "Percentage non-zero":nnz/total}
        # Backbone-agnostic concept-usage panel (contribution-mass effective counts,
        # participation ratio, NEC, per-image active concepts). W_g already has the arm-C
        # gate folded in, so no separate `gate=` here. val_c is standardised (dense), so
        # active_per_image_* is near the dictionary size and not very informative for LF-CBM.
        out_dict['concept_usage'] = concept_usage_report(W_g, val_c.cpu())
        out_dict['test_accuracy'] = test_acc
        if refit is not None:
            out_dict['refit'] = refit
        json.dump(out_dict, f, indent=2)
    
if __name__=='__main__':
    args = parser.parse_args()
    train_cbm_and_save(args)