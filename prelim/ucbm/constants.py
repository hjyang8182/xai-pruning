from os.path import join
import os 

BASE_PATH = join(os.path.dirname(os.path.abspath(__file__)), 'save')      
RESULT_PATH = join(BASE_PATH, 'RESULTS')

MODEL_DIR = {
    'cub_rn18':         join(BASE_PATH, 'CUB_RN18/'),
    # Reuse the checkpoint already downloaded under prelim/Label-free-CBM
    # instead of duplicating it under ucbm/save.
    'places365_rn18':   join(os.path.dirname(os.path.abspath(__file__)), '..', 'Label-free-CBM', 'data'),
    }

MODELS = ['resnet50_v2', 'cub_rn18', 'places365_rn18', 'clip_rn50']

# Shared raw-dataset cache under project/dataset, also used by prelim/Label-free-CBM,
# so datasets already downloaded there (cifar100, cub) don't need a second local copy.
DATA_ROOT = join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'dataset')

DATA_DIR = {
    'imagenet':     'data/ILSVRC2012/',
    'cub':          join(DATA_ROOT, 'CUB_200_2011/'),
    'places365':    join(DATA_ROOT, 'places365_torch'),
    'cifar100':     DATA_ROOT,
}

DATA_SETS = ['imagenet', 'imagenette', 'imagenet10', 'imagenet20', 'imagenet100', 
             'cub', 'places365', 'cifar100']
