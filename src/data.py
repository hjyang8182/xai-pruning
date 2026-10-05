import os
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.datasets.utils import download_and_extract_archive
from tqdm import tqdm
from src.config import DATA_PATH, BATCH_SIZE, SAE_BATCH, clip_device, device

# Preprocessing transformation for clip features
clip_preprocess = transforms.Compose([
    transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711]
    ),
])

# Load CIFAR100 dataset
def load_cifar100(data_root='./dataset'):
    train = datasets.CIFAR100(root=data_root, train=True,  download=True, transform=clip_preprocess)
    test = datasets.CIFAR100(root=data_root, train=False, download=True, transform=clip_preprocess)
    test_raw = datasets.CIFAR100(root=data_root, train=False, download=True)
    return train, test, test_raw

# CIFAR-100 fine label (0-99, matching torchvision.datasets.CIFAR100 class indices) -> coarse
# superclass label (0-19). Order taken from the official 'meta'/'train'/'test' pickles
# (dataset/cifar-100-python), which torchvision.datasets.CIFAR100 reads its labels from directly.
CIFAR100_COARSE_LABEL_NAMES = [
    'aquatic_mammals', 'fish', 'flowers', 'food_containers', 'fruit_and_vegetables',
    'household_electrical_devices', 'household_furniture', 'insects', 'large_carnivores',
    'large_man-made_outdoor_things', 'large_natural_outdoor_scenes', 'large_omnivores_and_herbivores',
    'medium_mammals', 'non-insect_invertebrates', 'people', 'reptiles', 'small_mammals', 'trees',
    'vehicles_1', 'vehicles_2',
]
CIFAR100_FINE_TO_COARSE = [
    4, 1, 14, 8, 0, 6, 7, 7, 18, 3, 3, 14, 9, 18, 7, 11, 3, 9, 7, 11,
    6, 11, 5, 10, 7, 6, 13, 15, 3, 15, 0, 11, 1, 10, 12, 14, 16, 9, 11, 5,
    5, 19, 8, 8, 15, 13, 14, 17, 18, 10, 16, 4, 17, 4, 2, 0, 17, 4, 18, 17,
    10, 3, 2, 12, 12, 16, 12, 1, 9, 19, 2, 10, 0, 1, 16, 12, 9, 13, 15, 13,
    16, 19, 2, 4, 6, 19, 5, 5, 8, 19, 18, 1, 2, 15, 6, 0, 17, 8, 14, 13,
]

def cifar100_fine_to_coarse(fine_labels):
    """Map CIFAR-100 fine (100-class) labels to their 20 coarse superclass labels."""
    lut = torch.tensor(CIFAR100_FINE_TO_COARSE, dtype=torch.long)
    return lut[fine_labels.long().cpu()]

# Load Food101 dataset
def load_food101(data_root='./dataset'):
    train = datasets.Food101(root=data_root, split='train',download=True, transform=clip_preprocess)
    test = datasets.Food101(root=data_root, split='test', download=True, transform=clip_preprocess)
    test_raw = datasets.Food101(root=data_root, split='test', download=True)
    return train, test, test_raw

# Load Oxford-IIIT Pet dataset
def load_oxford_pet(data_root='./dataset'):
    train = datasets.OxfordIIITPet(root=data_root, split='trainval', download=True, transform=clip_preprocess)
    test = datasets.OxfordIIITPet(root=data_root, split='test', download=True, transform=clip_preprocess)
    test_raw = datasets.OxfordIIITPet(root=data_root, split='test', download=True)
    return train, test, test_raw

# CUB-200-2011 (not built into torchvision.datasets, so downloads/parses manually)
class CUBDataset(Dataset):
    url = 'https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz'
    filename = 'CUB_200_2011.tgz'
    base_folder = 'CUB_200_2011'

    def __init__(self, root, train=True, transform=None, download=True):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.train = train

        if download and not os.path.isdir(os.path.join(self.root, self.base_folder)):
            download_and_extract_archive(self.url, self.root, filename=self.filename)

        self._load_metadata()

    def _load_metadata(self):
        base = os.path.join(self.root, self.base_folder)

        img_id_to_path = {}
        with open(os.path.join(base, 'images.txt')) as f:
            for line in f:
                img_id, path = line.strip().split(' ', 1)
                img_id_to_path[img_id] = path

        img_id_to_label = {}
        with open(os.path.join(base, 'image_class_labels.txt')) as f:
            for line in f:
                img_id, label = line.strip().split(' ')
                img_id_to_label[img_id] = int(label) - 1  # 0-indexed

        img_id_to_split = {}
        with open(os.path.join(base, 'train_test_split.txt')) as f:
            for line in f:
                img_id, is_train = line.strip().split(' ')
                img_id_to_split[img_id] = int(is_train)

        self.classes = []
        with open(os.path.join(base, 'classes.txt')) as f:
            for line in f:
                _, name = line.strip().split(' ', 1)
                self.classes.append(name.split('.', 1)[1].replace('_', ' '))

        want_split = 1 if self.train else 0
        self.samples = [
            (img_id_to_path[img_id], img_id_to_label[img_id])
            for img_id in img_id_to_path
            if img_id_to_split[img_id] == want_split
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(os.path.join(self.root, self.base_folder, 'images', path)).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label

def load_cub(data_root='./dataset'):
    train = CUBDataset(root=data_root, train=True,  download=True, transform=clip_preprocess)
    test = CUBDataset(root=data_root, train=False, download=True, transform=clip_preprocess)
    test_raw = CUBDataset(root=data_root, train=False, download=True)
    return train, test, test_raw

# Load Places365 dataset (small=True: 256x256 resized version, not the full-size ~160GB set)
def load_places365(data_root='./dataset'):
    train = datasets.Places365(root=data_root, split='train-standard', small=True, download=True, transform=clip_preprocess)
    test = datasets.Places365(root=data_root, split='val', small=True, download=True, transform=clip_preprocess)
    test_raw = datasets.Places365(root=data_root, split='val', small=True, download=True)
    return train, test, test_raw


# Save clip features for datasets
# TODO: Refactor to use acs GPU
def save_clip_features(clip_model, train_dataset, test_dataset, save_path):
    clip_model = clip_model.to(device)
    for split_name, dataset in [('train', train_dataset), ('test', test_dataset)]:
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
        all_features, all_labels = [], []

        with torch.no_grad():
            for imgs, labels in tqdm(loader, desc=split_name):
                feats = clip_model.encode_image(imgs.to(device)).float().cpu()
                all_features.append(feats)
                all_labels.append(labels)

        if device.type == 'mps':
            torch.mps.empty_cache()

        features = torch.cat(all_features)
        torch.save({'features': features, 'labels': torch.cat(all_labels)}, os.path.join(save_path, f'{split_name}_clip_features.pt'))
        print(f'Saved {split_name}: {features.shape}')

def load_clip_features(save_path, split):
    data = torch.load(os.path.join(save_path, f'{split}_clip_features.pt'), map_location='cpu')
    return data['features'], data['labels']

# Deterministic stratified subsample: return sorted indices keeping up to `per_class`
# examples of every label. Used to shrink Places365 (5000 imgs/class) for probe training
# without touching the cached full-set activations. Seeded independently of the training
# seeds so the same rows are held out on every run (and can be reproduced by the lambda
# sweeps if they take the same --subsample-per-class).
def subsample_per_class(labels, per_class, seed=42):
    labels = torch.as_tensor(labels)
    g = torch.Generator().manual_seed(seed)
    keep = []
    for c in labels.unique().tolist():
        idx = (labels == c).nonzero(as_tuple=True)[0]
        if idx.numel() > per_class:
            idx = idx[torch.randperm(idx.numel(), generator=g)[:per_class]]
        keep.append(idx)
    return torch.cat(keep).sort().values


# Get sparse activations from SAE encoder
def get_sae_acts(autoencoder, feats, device, batch_size=SAE_BATCH):
    autoencoder = autoencoder.to(device).eval()
    all_acts = []
    with torch.no_grad():
        for i in range(0, len(feats), batch_size):
            batch = feats[i:i+batch_size].unsqueeze(1).to(device)
            acts, _, *_ = autoencoder(batch)
            all_acts.append(acts.squeeze(1).cpu())
    return torch.cat(all_acts)

# Get activations from an untrained RandomConceptLayer, same batching convention as get_sae_acts
# (see src/models.py for why: the leakage-control counterpart to the real SAE encoder).
def get_random_concept_acts(random_layer, feats, device, batch_size=SAE_BATCH):
    random_layer = random_layer.to(device).eval()
    all_acts = []
    with torch.no_grad():
        for i in range(0, len(feats), batch_size):
            batch = feats[i:i+batch_size].to(device)
            all_acts.append(random_layer(batch).cpu())
    return torch.cat(all_acts)
