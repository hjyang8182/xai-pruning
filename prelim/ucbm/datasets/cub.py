from typing import Optional, Callable
from os import path
import pandas as pd
import torch

from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp",
                  ".pgm", ".tif", ".tiff", ".webp")

class CUB2011(ImageFolder):
    def __init__(self,
                 root: str,
                 train: bool,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 load_attributes: bool = False):
        '''
        load_attributes: bool = False
            If True, also load the per-image binary attribute (concept)
            labels from attributes/image_attribute_labels.txt into
            self.attributes, shape (len(self), #attributes), aligned
            with self.samples.
        '''

        # Download data set from 
        # https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz

        self.transform=transform
        self.target_transform=target_transform

        self.loader = default_loader
        self.extensions = IMG_EXTENSIONS

        # Load data from the folder. 
        images = pd.read_csv(path.join(root, 'images.txt'), sep=' ',
                             names=['img_id', 'image_path'])
        image_class_labels = \
            pd.read_csv(path.join(root, 'image_class_labels.txt'),
                        sep=' ', names=['img_id', 'target'])
        train_test_split = \
            pd.read_csv(path.join(root, 'train_test_split.txt'),
                        sep=' ', names=['img_id', 'is_training_img'])
        
        # Merge the data. 
        dataFrame = images.merge(image_class_labels, on='img_id')
        dataFrame = dataFrame.merge(train_test_split, on='img_id')

        # Move the dataclasses from 1 - 200 to 0 to 199
        dataFrame['target'] = dataFrame['target'].map(lambda x: x - 1)

        # Insert the beginning of the data path at the beginning. 
        dataFrame['image_path'] = dataFrame['image_path'].map(
            lambda x: path.join(root, 'images', x))
        
        # Split in training and test data
        if train:
            dataFrame = dataFrame[dataFrame.is_training_img == 1]
        else:
            dataFrame = dataFrame[dataFrame.is_training_img == 0]

        # Filter samples
        self.samples = [(row['image_path'], row['target'])
                        for _, row in dataFrame.iterrows()]
        self.targets = [s[1] for s in self.samples]

        # Load per-image binary attribute (concept) ground-truth labels.
        # A couple of rows in the raw file have a spurious extra column
        # (known CUB quirk for image_ids 2275/9364); sep=r'\s+' plus only
        # reading the first three columns sidesteps that.
        self.load_attributes = load_attributes
        if load_attributes:
            attr_labels = pd.read_csv(
                path.join(root, 'attributes', 'image_attribute_labels.txt'),
                sep=r'\s+', usecols=[0, 1, 2],
                names=['img_id', 'attribute_id', 'is_present'])
            attr_matrix = attr_labels.pivot(
                index='img_id', columns='attribute_id', values='is_present')
            attr_matrix = attr_matrix.reindex(dataFrame['img_id'])
            self.num_attributes = attr_matrix.shape[1]
            self.attributes = torch.tensor(
                attr_matrix.to_numpy(), dtype=torch.float32)

        # Load class labels.
        classes = pd.read_csv(path.join(root, 'classes.txt'), 
                              sep=' ', names=['class_id', 'class_label'])
        
        self.classes = [i.split('.')[-1] for i in classes['class_label']]
        self.class_to_idx = \
            {row['class_label'].split('.')[-1]: row['class_id'] - 1 
             for _, row in classes.iterrows()}