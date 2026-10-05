from src.config import *
import torch
import torch.nn as nn
from sparse_autoencoder import SparseAutoencoder
import clip
from gating import GatedProbe  # noqa: F401 (re-exported for src.models.GatedProbe callers)

class LinearProbe(nn.Module):
      def __init__(self, n_concepts, n_classes):
          super().__init__()
          self.linear = nn.Linear(n_concepts, n_classes)

      def forward(self, x):
          return self.linear(x)

class RandomConceptLayer(nn.Module):
    """Leakage-control stand-in for the trained SAE: same Linear+ReLU shape as the real encoder
    (see LinearEncoder in Discover-then-Name), but weights are drawn once from a Gaussian
    (fan-in scaled, as for a ReLU layer) and frozen - never fit to data. If a probe trained on top
    of this does about as well as one trained on real SAE concepts, the real probe's accuracy is
    coming from having enough dimensions to fit any linear function of CLIP space, not from what
    the concepts encode."""
    def __init__(self, n_input_features, n_learned_features, seed=0):
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        std = (2.0 / n_input_features) ** 0.5
        weight = torch.randn(n_learned_features, n_input_features, generator=generator) * std
        self.linear = nn.Linear(n_input_features, n_learned_features)
        with torch.no_grad():
            self.linear.weight.copy_(weight)
            self.linear.bias.zero_()
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        return torch.relu(self.linear(x))

# class UCBMProbe(nn.Mod)
def load_autoencoder(device):
    autoencoder = SparseAutoencoder(
        n_input_features=N_INPUT_FEATURES,
        n_learned_features=N_LEARNED_FEATURES,
        n_components=N_COMPONENTS,
    )
    state_dict = torch.load(CHECKPOINT_PATH, map_location='cpu')
    autoencoder.load_state_dict(state_dict)
    return autoencoder.to(device).eval()

def load_clip():
    model, preprocess = clip.load("RN50", device=clip_device)
    for param in model.parameters():
        param.requires_grad = False
    return model.eval(), preprocess
