import argparse
import json
import os
import random
from datetime import datetime
import torch
from src import config
from src.models import GatedProbe, LinearProbe
from src.concepts import load_concept_names
from src.data import load_cifar100, load_food101, load_cub, load_oxford_pet, load_places365
from src.train import concept_contributions
from src.visualise import (explain_image, n_available_concepts, plot_concept_explanation,
                           plot_concept_explanation_grid, plot_contribution_distribution)

DATASET_LOADERS = {'cifar100': load_cifar100, 'food101': load_food101, 'cub': load_cub, 'oxford_pet': load_oxford_pet, 'places365': load_places365}
N_CLASSES       = {'cifar100': 100, 'food101': 101, 'cub': 200, 'oxford_pet': 37, 'places365': 365}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Explain predictions on a random sample of test images by showing each probe's "
                    "top-k contributing concepts (activation x final-layer weight for the predicted "
                    "class), with the remaining concepts' contributions summed into a single "
                    "'Sum of other features' bar."
    )
    parser.add_argument('-d', '--dataset', choices=DATASET_LOADERS.keys(), default='oxford_pet')
    parser.add_argument('--gated-run', required=True,
                         help='data/<dataset>/model/<run> holding the gated probe checkpoint to explain')
    parser.add_argument('--baseline-run', default=None,
                         help='data/<dataset>/model/<run> holding the baseline probe checkpoint to explain '
                              'alongside the gated one; omit to only explain the gated probe')
    parser.add_argument('--probe-seed', type=int, default=0,
                         help='Which probe_seed<N>.pt (or probe.pt) checkpoint to load from each run')
    parser.add_argument('--n-images', type=int, default=5, help='Number of random test images to explain')
    parser.add_argument('--top-k', type=int, default=5, help='Number of top contributing concepts to show per image')
    parser.add_argument('--image-seed', type=int, default=0, help='Seed for randomly sampling test images')
    parser.add_argument('--images', type=int, nargs='+', default=None,
                         help='Explicit test-image indices to explain instead of a random sample (e.g. to '
                              'hand-pick the examples that go in the paper)')
    parser.add_argument('--refit', action='store_true',
                         help='Explain the gated run\'s mask-and-refit head (probe_seed<N>_refit.pt: the open '
                              'concepts as a hard mask with the linear head refit on them) instead of the '
                              'soft/STE gated probe. This is the deliverable train_cbm.py reports as "refit" accuracy.')
    parser.add_argument('--select', choices=['random', 'gated-wins', 'baseline-wins', 'both-right', 'both-wrong'],
                         default='random',
                         help='Which test images the random sample is drawn from: any (random), those the gated '
                              'probe classifies correctly and the baseline does not (gated-wins), the reverse '
                              '(baseline-wins), or those both get right / wrong. Needs --baseline-run except for '
                              '"random". Ignored when --images is given.')
    parser.add_argument('--grid', action='store_true',
                         help='Also stack every explained image into one figure (a row per image), '
                              'concept_explanation_grid_<stamp>.svg')
    return parser.parse_args()


# A run dir holds one probe_seed<seed>.pt per seed trained by train_cbm.py; older runs may only
# have a single probe.pt.
def _probe_path(run_dir, seed):
    seed_path = os.path.join(run_dir, f'probe_seed{seed}.pt')
    if os.path.exists(seed_path):
        return seed_path
    single_path = os.path.join(run_dir, 'probe.pt')
    if os.path.exists(single_path):
        return single_path
    raise FileNotFoundError(f'No probe_seed{seed}.pt or probe.pt found in {run_dir}')


def main(args):
    dataset   = args.dataset
    n_classes = N_CLASSES[dataset]
    DATA_DIR      = os.path.join(config.DATA_PATH, dataset)
    EXPLAIN_DIR   = os.path.join(DATA_DIR, 'explanations')
    ACT_SAVE  = os.path.join(DATA_DIR, 'activations')
    MODEL_DIR = os.path.join(DATA_DIR, 'model')
    os.makedirs(EXPLAIN_DIR, exist_ok=True)

    test_acts_path = os.path.join(ACT_SAVE, 'test_sae_acts.pt')
    if not os.path.exists(test_acts_path):
        raise FileNotFoundError(f'{test_acts_path} not found; run train_cbm.py for this dataset first')
    test_acts = torch.load(test_acts_path)

    # test_raw holds un-normalized images (for display) in the same row order as test_acts.
    _, _, test_raw = DATASET_LOADERS[dataset]()
    class_names   = test_raw.classes
    concept_names = load_concept_names()

    gated_run_dir = args.gated_run if os.path.isabs(args.gated_run) else os.path.join(MODEL_DIR, args.gated_run)
    if args.refit:
        # Refit head: a plain linear layer whose weight is exactly zero outside the kept concepts.
        # `keep` is attached so n_available_concepts counts the mask rather than all 8192.
        refit = torch.load(os.path.join(gated_run_dir, f'probe_seed{args.probe_seed}_refit.pt'), map_location='cpu')
        gated_probe = LinearProbe(config.N_LEARNED_FEATURES, n_classes)
        with torch.no_grad():
            gated_probe.linear.weight.copy_(refit['weight'])
            gated_probe.linear.bias.copy_(refit['bias'])
        gated_probe.keep = refit['keep'].bool()
    else:
        gated_probe = GatedProbe(config.N_LEARNED_FEATURES, n_classes)
        gated_probe.load_state_dict(torch.load(_probe_path(gated_run_dir, args.probe_seed), map_location='cpu'))
    gated_probe.eval()
    probes = {'Gated': gated_probe}

    if args.baseline_run:
        baseline_run_dir = args.baseline_run if os.path.isabs(args.baseline_run) else os.path.join(MODEL_DIR, args.baseline_run)
        baseline_probe = LinearProbe(config.N_LEARNED_FEATURES, n_classes)
        baseline_probe.load_state_dict(torch.load(_probe_path(baseline_run_dir, args.probe_seed), map_location='cpu'))
        baseline_probe.eval()
        probes = {'Baseline': baseline_probe, 'Gated': gated_probe}

    if args.images:
        img_indices = args.images
    else:
        pool = range(len(test_raw))
        if args.select != 'random':
            if not args.baseline_run:
                raise ValueError(f'--select {args.select} needs --baseline-run')
            targets = getattr(test_raw, 'targets', None) or getattr(test_raw, 'labels', None)
            if targets is None:  # fall back to decoding every image (slow on the big datasets)
                targets = [test_raw[i][1] for i in range(len(test_raw))]
            labels = torch.as_tensor(list(targets))
            with torch.no_grad():
                right = {}
                for label, probe in probes.items():
                    out = probe(test_acts)
                    logits = out[0] if isinstance(out, tuple) else out
                    right[label] = logits.argmax(dim=1) == labels
            want = {'gated-wins': right['Gated'] & ~right['Baseline'],
                    'baseline-wins': right['Baseline'] & ~right['Gated'],
                    'both-right': right['Gated'] & right['Baseline'],
                    'both-wrong': ~right['Gated'] & ~right['Baseline']}[args.select]
            pool = want.nonzero(as_tuple=True)[0].tolist()
            print(f'--select {args.select}: {len(pool)} of {len(test_raw)} test images qualify')
        rng = random.Random(args.image_seed)
        img_indices = rng.sample(list(pool), min(args.n_images, len(pool)))

    results = []
    for img_idx in img_indices:
        image, y_true = test_raw[img_idx]
        acts = test_acts[img_idx]
        # The display image is saved next to the results so cross-dataset figures
        # (plot_explanation_examples.py) can be drawn from the JSON without reloading datasets.
        image_path = os.path.join(EXPLAIN_DIR, f'img{img_idx}.png')
        if not os.path.exists(image_path):
            image.save(image_path)
        entry = {'img_idx': img_idx, 'true_class': class_names[y_true], 'image_path': image_path}
        print(f'\nImage {img_idx}  true={entry["true_class"]}')

        for label, probe in probes.items():
            pred, top, other = explain_image(probe, acts, concept_names, y_true, class_names, args.top_k)
            entry[label.lower()] = {
                'predicted_class': class_names[pred],
                'correct': pred == y_true,
                'top_concepts': [{'concept': n, 'contribution': v} for n, v in top],
                'sum_of_other_features': other,
                'n_concepts_available': n_available_concepts(probe, len(concept_names)),
            }
            print(f'  {label} -> pred={class_names[pred]}')
            for n, v in top:
                print(f'    {n}: {v:.4f}')
            print(f'    Sum of other features: {other:.4f}')

        plot_path = plot_concept_explanation(
            img_idx, probes, acts, concept_names, class_names, test_raw, dataset, top_k=args.top_k,
            explain_dir=EXPLAIN_DIR,
        )
        entry['plot_path'] = plot_path

        # Distribution of |contribution| across this image's concepts, per probe - shows how
        # much sparser the gated probe's explanation is than the baseline's for this prediction.
        contribs_by_label = {label: concept_contributions(probe, acts.unsqueeze(0))[1].abs().flatten().cpu().numpy()
                              for label, probe in probes.items()}
        dist_plot_path = plot_contribution_distribution(
            contribs_by_label, dataset, save_dir=EXPLAIN_DIR,
            plot_label=f'contribution_distribution_img{img_idx}', title_suffix=f' (image {img_idx})',
        )
        entry['contribution_distribution_plot_path'] = dist_plot_path
        results.append(entry)

    grid_path = None
    if args.grid:
        grid_path = plot_concept_explanation_grid(
            img_indices, probes, test_acts, concept_names, class_names, test_raw, dataset,
            top_k=args.top_k, explain_dir=EXPLAIN_DIR)
        print(f'\nSaved combined figure to {grid_path}')

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_path = os.path.join(EXPLAIN_DIR, f'concept_explanations_{timestamp}.json')
    with open(results_path, 'w') as f:
        json.dump({
            'dataset': dataset, 'gated_run': args.gated_run, 'baseline_run': args.baseline_run,
            'probe_seed': args.probe_seed, 'refit': args.refit, 'select': args.select, 'top_k': args.top_k, 'image_seed': args.image_seed,
            'grid_plot_path': grid_path, 'images': results,
        }, f, indent=2)
    print(f'\nSaved results to {results_path}')


if __name__ == '__main__':
    main(parse_args())
