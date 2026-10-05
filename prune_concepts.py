import argparse
import os
import numpy as np
import torch
from src import config
from src.models import load_autoencoder, load_clip
from src.data import load_cifar100, load_food101, load_cub, load_oxford_pet, save_clip_features, load_clip_features, get_sae_acts
from src.concepts import load_vocab, save_vocab_embeddings, load_vocab_embeddings, compute_pruning_criteria
from src.train import train_probe
from src.visualise import plot_pruning_results

DATASET_LOADERS = {'cifar100': load_cifar100, 'food101': load_food101, 'cub': load_cub, 'oxford_pet': load_oxford_pet}
N_CLASSES       = {'cifar100': 100, 'food101': 101, 'cub': 200, 'oxford_pet': 37}
KEEP_FRACTIONS  = {'cifar100': config.KEEP_FRACTIONS_CIFAR, 'food101': config.KEEP_FRACTIONS_FOOD, 'cub': config.KEEP_FRACTIONS_CUB, 'oxford_pet': config.KEEP_FRACTIONS_OXFORD_PET}


def parse_args():
    parser = argparse.ArgumentParser(description='Prune concepts by criteria and plot accuracy vs fraction of concepts kept')
    parser.add_argument('-d', '--dataset', choices=DATASET_LOADERS.keys(), default='cifar100')
    parser.add_argument('--lr', type=float, default=config.LEARNING_RATE)
    parser.add_argument('--epochs', type=int, default=config.EPOCHS)
    parser.add_argument('--batch-size', type=int, default=config.BATCH_SIZE)
    parser.add_argument('--lambda-sparse', type=float, default=config.LAMBDA_SPARSE)
    return parser.parse_args()


def main(args):
    dataset        = args.dataset
    n_classes      = N_CLASSES[dataset]
    keep_fractions = KEEP_FRACTIONS[dataset]
    SAVE           = os.path.join(config.DATA_PATH, dataset)
    ACT_SAVE       = os.path.join(SAVE, 'activations')
    os.makedirs(ACT_SAVE, exist_ok=True)

    train_dataset, test_dataset, _ = DATASET_LOADERS[dataset]()

    if not os.path.exists(os.path.join(ACT_SAVE, 'train_clip_features.pt')):
        clip_model, _ = load_clip()
        save_clip_features(clip_model, train_dataset, test_dataset, ACT_SAVE)
    train_feats, train_labels = load_clip_features(ACT_SAVE, 'train')
    test_feats,  test_labels  = load_clip_features(ACT_SAVE, 'test')

    autoencoder = load_autoencoder(config.sae_device)

    train_acts_path = os.path.join(ACT_SAVE, 'train_sae_acts.pt')
    test_acts_path  = os.path.join(ACT_SAVE, 'test_sae_acts.pt')
    if not os.path.exists(train_acts_path):
        train_acts = get_sae_acts(autoencoder, train_feats, config.sae_device)
        test_acts  = get_sae_acts(autoencoder, test_feats,  config.sae_device)
        torch.save(train_acts, train_acts_path)
        torch.save(test_acts,  test_acts_path)
    train_acts = torch.load(train_acts_path)
    test_acts  = torch.load(test_acts_path)

    if not os.path.exists(config.EMB_PATH):
        clip_model, _ = load_clip()
        save_vocab_embeddings(clip_model, load_vocab())
    text_emb = load_vocab_embeddings()

    criteria_path = os.path.join(ACT_SAVE, 'criteria.pt')
    if not os.path.exists(criteria_path):
        torch.save(compute_pruning_criteria(train_acts, autoencoder, text_emb), criteria_path)
    criteria = torch.load(criteria_path)

    base_acc, _ = train_probe(
        train_acts, train_labels, test_acts, test_labels, n_classes=n_classes,
        lr=args.lr, epochs=args.epochs, batch_size=args.batch_size, lambda_sparse=args.lambda_sparse,
    )
    print(f'Baseline accuracy (all {config.N_LEARNED_FEATURES} concepts): {base_acc:.4f}')

    results_path = os.path.join(ACT_SAVE, 'pruning_results.pt')
    if not os.path.exists(results_path):
        results = {}
        for name, scores in criteria.items():
            ranking = np.argsort(scores)
            accs = [train_probe(train_acts[:, ranking[-int(f * config.N_LEARNED_FEATURES):]],
                                train_labels,
                                test_acts[:,  ranking[-int(f * config.N_LEARNED_FEATURES):]],
                                test_labels, n_classes=n_classes,
                                lr=args.lr, epochs=args.epochs, batch_size=args.batch_size,
                                lambda_sparse=args.lambda_sparse)[0]
                    for f in keep_fractions]
            results[name] = (keep_fractions, accs)
        torch.save(results, results_path)
    results = torch.load(results_path)

    plot_pruning_results(results, base_acc, keep_fractions, f'DN-CBM Accuracy vs Fraction of Concepts Kept — {dataset}', dataset)


if __name__ == '__main__':
    main(parse_args())
