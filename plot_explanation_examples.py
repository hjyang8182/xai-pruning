"""One explanation example per dataset in a single figure (a row per dataset: test image, then
the baseline's and the gated probe's top-k contributing concepts), for showing side by side how
the two probes explain the same prediction across datasets.

Reads the newest concept_explanations_*.json under data/<dataset>/explanations/ (written by
explain_concepts.py, which also saves each explained image next to it) and takes one image from
each: by default the first correctly-classified-by-both image, or --images to pick test indices.
"""
import argparse
import glob
import json
import os

from src.config import DATA_PATH
from src.visualise import plot_explanation_examples

DISPLAY_NAMES = {'cifar100': 'CIFAR-100', 'cub': 'CUB', 'places365': 'Places365', 'imagenet': 'ImageNet', 'food101': 'Food-101', 'oxford_pet': 'Oxford-IIIT Pet'}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--datasets', nargs='+', default=['cifar100', 'cub', 'places365'])
    parser.add_argument('--json', nargs='*', default=None,
                        help='explicit concept_explanations json per dataset (same order as --datasets); '
                             'default: newest under data/<dataset>/explanations/')
    parser.add_argument('--images', type=int, nargs='*', default=None,
                        help='test-image index to show per dataset (same order as --datasets; must be one '
                             'of the images in that JSON). Default: first image both probes got right, '
                             'else the first image.')
    return parser.parse_args()


def _latest_json(dataset):
    paths = sorted(glob.glob(os.path.join(DATA_PATH, dataset, 'explanations', 'concept_explanations_*.json')))
    if not paths:
        raise FileNotFoundError(f'no concept_explanations_*.json under data/{dataset}/explanations; '
                                f'run explain_concepts.py -d {dataset} --baseline-run ... --gated-run ... first')
    return paths[-1]


def _pick(entries, img_idx):
    if img_idx is not None:
        matches = [e for e in entries if e['img_idx'] == img_idx]
        if not matches:
            raise ValueError(f'image {img_idx} not in this JSON (has {[e["img_idx"] for e in entries]})')
        return matches[0]
    both_right = [e for e in entries if e['baseline'].get('correct') and e['gated'].get('correct')]
    return (both_right or entries)[0]


def main(args):
    paths = args.json or [_latest_json(d) for d in args.datasets]
    picks = args.images or [None] * len(args.datasets)
    if len(paths) != len(args.datasets) or len(picks) != len(args.datasets):
        raise ValueError('--json / --images need one entry per dataset')

    examples = []
    for dataset, path, img_idx in zip(args.datasets, paths, picks):
        with open(path) as f:
            results = json.load(f)
        if 'baseline' not in results['images'][0] or 'image_path' not in results['images'][0]:
            raise ValueError(f'{path} lacks a baseline probe or saved images; re-run explain_concepts.py '
                             f'-d {dataset} with --baseline-run')
        entry = _pick(results['images'], img_idx)
        print(f'{dataset}: {path} -> image {entry["img_idx"]} ({entry["true_class"]})')
        examples.append((DISPLAY_NAMES.get(dataset, dataset), entry))

    out = plot_explanation_examples(examples)
    print(f'Saved plot to {out}')


if __name__ == '__main__':
    main(parse_args())
