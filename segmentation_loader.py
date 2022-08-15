''' 
Custom DataLoader for segemtation tasks
'''

import os
import random
from glob import glob
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from albumentations.augmentations.transforms import RandomShadow


# helper functions
def mask2cat(mask):
    ''' Converts the RGB mask to a categorical image
        suitable for training the network

        red (drivable) == 0
        green (background) == 1
        blue (adjacent) == 2
        '''
    mask_red = np.zeros(mask.shape[:2]) 
    mask_green = np.logical_or(mask_red, mask[:, :, 1]).astype(np.float32)
    mask_blue = np.logical_or(mask_red, mask[:, :, 2]).astype(np.float32)*2

    mask_out = (mask_green + mask_blue).astype(np.uint8)

    return mask_out

def tensor_mask2cat(mask):
    ''' Converts the RGB mask to a categorical image
        suitable for training the network

        red (drivable) == 0
        green (background) == 1
        blue (adjacent) == 2
        '''
    mask_red = torch.zeros(mask.shape[1:]) 
    mask_green = torch.logical_or(mask_red, mask[1, :, :]).to(torch.float32)
    mask_blue = torch.logical_or(mask_red, mask[2, :, :]).to(torch.float32)*2

    mask_out = (mask_green + mask_blue).to(torch.uint8)

    return mask_out
    

def cat2mask(mask_cat):
    ''' reverses the mapping from category to RGB mask '''
    mask_out = np.zeros(tuple(list(mask_cat.shape) + [3]))

    mask_out[mask_cat == 0, 0] = 1 # red (drivable)
    mask_out[mask_cat == 1, 1] = 1 # green (background)
    mask_out[mask_cat == 2, 2] = 1 # blue (adjacent)

    # convert to uint8
    mask_out = (mask_out*255).astype(np.uint8)

    return mask_out




class SegmentationDataset(Dataset):

    def __init__(self, data_path=None, transforms=None, 
                 normalize_images=True, extra_transforms=True):
        super().__init__()

        self.transforms = transforms
        self.extra_transforms = extra_transforms # extra transforms for images only (not masks)
        self.normalize_images = normalize_images

        self.image_paths = glob(os.path.join(data_path, 'images/*.png'))
        self.mask_paths = glob(os.path.join(data_path, 'masks/*.png'))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        mask_path = self.mask_paths[index]

        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path)

        # resize masks to match network output
        # if self.mask_resize:
        #     mask = cv2.resize(mask, (self.mask_resize))

        if self.transforms:
            # apply the same random transforms to the image and mask

            # get current random state before first transform
            state = torch.get_rng_state() 
            image = self.transforms(image)
            
            # reset random state to that of the previous transform
            torch.set_rng_state(state) 
            mask = self.transforms(mask)

            # round all mask values to the nearest 0 or 1
            mask = torch.round(mask, decimals=0)

        if self.extra_transforms:
            jitter = transforms.ColorJitter((0.85, 1.15), 
                                            (0.85, 1.15), 
                                            (0.85, 1.15), 
                                            (0.1, 0.1))
            image = jitter(image)

            # augment random shadows
            image = RandomShadowTransform()(image)

        # normalize images
        if self.normalize_images:
            normalize = transforms.Normalize(
                                    [0.485, 0.456, 0.406], 
                                    [0.229, 0.224, 0.225])
            image = normalize(image)

        # convert mask to a 1 Dimensional categorical
        mask = tensor_mask2cat(mask)

        return image, mask


class RandomShadowTransform(object):
    def __init__(self, 
                 shadow_roi=(0, 0, 1, 1), 
                 num_shadows_lower=2, 
                 num_shadows_upper=4, 
                 shadow_dimension=5, 
                 always_apply=False, 
                 p=0.75):
        
        self.shadow_roi = shadow_roi
        self.num_shadows_lower = num_shadows_lower 
        self.num_shadows_upper = num_shadows_upper 
        self.shadow_dimension = shadow_dimension 
        self.always_apply = always_apply 
        self.p = p

    def __call__(self, image):
        random_shadow = RandomShadow(
                                self.shadow_roi,
                                self.num_shadows_lower,
                                self.num_shadows_upper,
                                self.shadow_dimension,
                                self.always_apply,
                                self.p)
        shadowed = random_shadow(image=image.permute(1, 2, 0))['image']

        # sometimes it doesn't return a tensor??
        if torch.is_tensor(shadowed):
            shadowed = shadowed.permute(2, 0, 1)
        else:
            shadowed = transforms.ToTensor()(shadowed)

        return shadowed




