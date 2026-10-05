import os
import torch

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project/

CHECKPOINT_PATH   = os.path.join(BASE_DIR, 'Discover-then-Name', 'checkpoints', 'clip_RN50_sparse_autoencoder_final.pt')
DATA_PATH = os.path.join(BASE_DIR, 'data')
VOCAB_PATH        = os.path.join(BASE_DIR, 'Discover-then-Name', 'vocab', 'clipdissect_20k.txt')
EMB_PATH          = os.path.join(BASE_DIR, 'Discover-then-Name', 'vocab', 'embeddings_clip_RN50_clipdissect_20k.pth')
CSV_PATH          = os.path.join(BASE_DIR, 'Discover-then-Name', 'vocab', 'concept_names.csv')

# --- Devices ---
device      = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
sae_device  = device
clip_device = device

# --- SAE / model dims ---
N_INPUT_FEATURES   = 1024
N_LEARNED_FEATURES = 8192
N_COMPONENTS       = 1

# --- Data loading ---
BATCH_SIZE  = 64          # image batch for CLIP feature extraction (224x224 imgs)
SAE_BATCH   = 1024
VOCAB_BATCH = 512

# --- Training defaults ---
LEARNING_RATE  = 1e-3
EPOCHS         = 100
# Minibatch for the linear/gated probe, which trains on cached concept activations (not
# images). Far larger than BATCH_SIZE: it's a single Linear layer, so big batches keep the
# GPU busy and cut the Python/kernel-launch overhead that dominates at batch 64.
PROBE_BATCH_SIZE = 4096
LAMBDA_SPARSE  = 1e-4
LAMBDA_GATE    = 1e-4

# --- Pruning ---
KEEP_FRACTIONS_CIFAR = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
KEEP_FRACTIONS_FOOD  = [0.1, 0.3, 0.5, 0.7, 0.9]
KEEP_FRACTIONS_CUB   = [0.1, 0.3, 0.5, 0.7, 0.9]
KEEP_FRACTIONS_OXFORD_PET = [0.1, 0.3, 0.5, 0.7, 0.9]
PRUNE_FRACTION = 0.5

# --- Reproducibility ---
SEED = 42
