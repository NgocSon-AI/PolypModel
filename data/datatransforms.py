import torch
import os
import torchvision.transforms.functional as F
from PIL import Image
import numpy as np

def default_augment_seg(input_image, input_mask):
    input_image = F.adjust_brightness(input_image, brightness_factor=0.1)
    input_image = F.adjust_contrast(input_image, contrast_factor=torch.FloatTensor([0.9, 1.1]).uniform_().item())
    input_image = F.adjust_saturation(input_image, saturation_factor=torch.FloatTensor([0.9, 1.1]).uniform_().item())
    input_image = F.adjust_hue(input_image, hue_factor=torch.FloatTensor([-0.01, 0.01]).uniform_().item())

    return input_image, input_mask

def BatchAdvAugmentSeg(imagesT, masksT):
    images, masks = default_augment_seg(imagesT, masksT)
    return images, masks

def build_decoder(ext, ext2, with_labels=True, target_size=(256, 256), segment=False):
    def decode(path):
        img = Image.open(path)
        if ext in ['png', 'jpg', 'jpeg']:
            img = img.convert('RGB')
        else:
            raise ValueError("Image extension not supported ")
        img = F.resize(F.to_tensor(img), target_size)
        img = torch.Tensor(np.array(img)).permute(2, 0, 1)
        return img
    
    def decode_mask(path, gray=True):
        img = Image.open(path)
        if ext2 in ['png', 'jpg', 'jpeg']:
            img = img.convert('RGB')
        else:
            raise ValueError('Image extension not supported')
        if gray:
            img = img.convert('L')
        img = F.resize(F.to_tensor(img), target_size)
        img = img.float() / 255.0
        img = torch.Tensor(np.array(img)).permute(2, 0, 1)
        return img
    
    def decode_with_labels(path, label):
        return decode(path), label

    def decode_with_segments(path, path2, gray=True):
        return decode(path), decode_mask(path2, gray)
    if segment:
        return decode_with_segments

    return decode_with_labels if with_labels else decode

def build_augmenter(with_labels=True):
    def augment(img):
        img = torch.flip(img, dims=[-1])
        img = torch.flip(img, dims=[-2])

        img = F.adjust_brightness(img, brightness_factor=0.1)
        img = F.adjust_contrast(img, contrast_factor=torch.FloatTensor([0.9, 1.1]).uniform_().item())
        img = F.adjust_saturation(img, saturation_factor=torch.FloatTensor([0.9, 1.1]).uniform_().item())
        img = F.adjust_hue(img, hue_factor=torch.FloatTensor([-0.01, 0.01]).uniform_().item())
        return img

    def augment_with_labels(img, label):
        return augment(img), label

    return augment_with_labels if with_labels else augment