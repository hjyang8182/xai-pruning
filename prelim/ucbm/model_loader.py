from typing import Callable, Literal, Union
from os import path
from constants import MODEL_DIR

import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights, resnet18
import torchvision.transforms as transforms
from pytorchcv.model_provider import get_model as ptcv_get_model
import open_clip


class Model:
    '''
    Dataclass managing machine learning models. 

    Parameters
    ----------
    model: Callable | torch.nn.Module
        Machine learning model. 
    g: Callable | torch.nn.model
        First part of the model split. 
    h: Callable | torch.nn.model
        Second part of the model split. 
    transform: Callable
        The function that is applied on RGB-PIL-Image to get
        a valid model input. 
    '''

    def __init__(self, 
                 model: Union[Callable, nn.Module], 
                 g: Union[Callable, nn.Module], 
                 h: Union[Callable, nn.Module], 
                 transform: Callable):
        self.model = model
        self.g = g
        self.h = h
        self.transform = transform


################################################################################
#------------------------------------------------------------------------------#
#---------------------------------Model-Loader---------------------------------#
#------------------------------------------------------------------------------#
################################################################################

        
def get_resnet_imagenet_preprocess():
    target_mean = [0.485, 0.456, 0.406]
    target_std = [0.229, 0.224, 0.225]
    preprocess = transforms.Compose(
        [transforms.Resize(256), 
            transforms.CenterCrop(224),
            transforms.ToTensor(), 
            transforms.Normalize(mean=target_mean, std=target_std)])
    return preprocess

def get_model(model_name: str, 
              device: Literal['cpu', 'cuda'], 
              **kwargs) -> Model:
    '''
    Load a model to the given device. 

    Parameters
    ----------
    model_name: str
        The name of the model. 
    device: Literal['cpu', 'cuda']
        The device to load the model to. 
    '''

    if model_name == 'resnet50_v2':
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        model = model.eval().to(device)

        g = nn.Sequential(*(list(model.children())[:-2]))
        def h(x):
            children = list(model.children())
            x = children[-2](x)
            x = torch.flatten(x, 1)
            x = children[-1](x)
            return x
        
        transform = ResNet50_Weights.IMAGENET1K_V2.transforms()

        return Model(model, g, h, transform)

    elif model_name == 'cub_rn18':
        model = ptcv_get_model("resnet18_cub", pretrained=True, 
                               root=MODEL_DIR['cub_rn18'])
        model = model.eval().to(device)

        g = nn.Sequential(*(list(list(model.children())[0].children())[:-1]))
        def h(x):
            children = list(model.children())
            out = list(children[0])[-1](x)
            out = torch.flatten(out, 1)
            out = nn.Sequential(*(children[1:]))(out)
            return out
        
        return Model(model, g, h, get_resnet_imagenet_preprocess())
    

    elif model_name == 'places365_rn18':
        model = resnet18(pretrained=False, num_classes=365).to(device)
        f_path = path.join(MODEL_DIR['places365_rn18'], 
                           'resnet18_places365.pth.tar')
        state_dict = torch.load(f_path)['state_dict']
        new_state_dict = {}
        for key in state_dict:
            if key.startswith('module.'):
                new_state_dict[key[7:]] = state_dict[key]
        model.load_state_dict(new_state_dict)
        model.eval()

        g = nn.Sequential(*(list(model.children())[:-1]))
        def h(x):
            x = torch.flatten(x, 1)
            x = list(model.children())[-1](x)
            return x

        transform = get_resnet_imagenet_preprocess()

        return Model(model, g, h, transform)

    elif model_name == 'clip_rn50':
        # OpenAI's original CLIP RN50 weights, same backbone LF-CBM/VLG-CBM call "clip_RN50".
        clip_model, _, preprocess = open_clip.create_model_and_transforms(
            'RN50', pretrained='openai', device=device)
        clip_model = clip_model.eval()
        visual = clip_model.visual

        # CRAFT (Craft.fit_all in Craft/craft/craft_torch.py) calls
        # `self.input_to_latent(patches).mean(dim=(2,3))` unconditionally on `g`'s output, and
        # `Craft.fit` additionally asserts activations are non-negative (required by NMF). CLIP's
        # own `encode_image` already pools spatially via attention down to (B, embed_dim), which
        # both breaks the `dim=(2,3)` mean (no spatial dims left) and isn't guaranteed
        # non-negative (no final ReLU). So `g` here instead runs the visual trunk only up through
        # `layer4` -- CLIP RN50 is architecturally a standard ResNet50 (stem -> layer1..4) with an
        # attention-pool head bolted on instead of avgpool+fc -- giving a (B, 2048, 7, 7),
        # ReLU-terminated (non-negative) spatial map, matching resnet50_v2's `g` shape/semantics
        # exactly. ucbm.py's own `predict()` mean-pools this the same way resnet50_v2's output is
        # pooled, so classifier training still works unchanged.
        def g(x):
            x = visual.stem(x)
            x = visual.layer1(x)
            x = visual.layer2(x)
            x = visual.layer3(x)
            x = visual.layer4(x)
            return x

        def h(x):
            raise NotImplementedError(
                "clip_rn50 has no classification head to split off -- CLIP is trained "
                "contrastively, not as an image classifier. `h` is unused by both "
                "train_cbm.py and by Craft.fit/fit_all/transform (latent_to_logit is stored "
                "but never called in this repo's craft_torch.py)."
            )

        return Model(clip_model, g, h, preprocess)

    raise AttributeError(f"Model {model_name} is not provided. ")