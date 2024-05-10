from enum import Flag, auto
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
from torch.utils.data import Dataset
from torchvision.datasets import MNIST

torchvision.disable_beta_transforms_warning()
from torchvision.transforms import v2

class DataMode(Flag):
    MONO = 0
    COLOR = auto()
    BINARY = auto()
    MISSING = auto()

class Binarize(nn.Module):
    def __init__(self, threshold=0.5):
        super().__init__()
        self.threshold = threshold

    def forward(self, data):  
        data.data = (data.data > self.threshold).to(data.data.dtype)
        return data

class OneHot(nn.Module):
    def forward(self, data):
        data = torch.Tensor([data])
        out = nn.functional.one_hot((data % 10).to(torch.int64), num_classes=10).to(torch.float32)
        return out.reshape(10)


class StackedMNIST(Dataset):
    def __init__(
        self,
        root: str | Path = Path(''),
        mode: DataMode = DataMode.MONO | DataMode.BINARY,
        train: bool = True,
    ):  
        super().__init__()
        self.mode = mode

        transforms = [ v2.ToTensor(), v2.ToDtype(torch.float32, scale=True) ]
        target_transforms = [ v2.ToTensor() ]

        if (DataMode.MONO not in self.mode) and (DataMode.COLOR not in self.mode):
            mode |= DataMode.MONO
            
        if DataMode.BINARY in mode:
            transforms.append(Binarize())

        self.transforms = v2.Compose(transforms)
        self.target_transforms = v2.Compose(target_transforms)

        self.data = MNIST(
            root,
            train=train,
            download=True,
            transform=(self.transforms),
            target_transform=(self.target_transforms) 
        )
    
    def __getitem__(self, index):
        data = self.transforms(self.data.data[[index]])
        label = self.target_transforms(self.data.targets[index])
        target = OneHot()(label)

        if (DataMode.MISSING in self.mode) and label == 8:
            new_index = np.random.randint(len(self.data.data))
            data, target, label = self._get_random_item(new_index)

        if (DataMode.COLOR in self.mode):
            gb_indices = np.random.randint(len(self.data.data), size=(2,))

            green_image, green_target, green_label = self._get_random_item(gb_indices[0])
            blue_image, blue_target, blue_label = self._get_random_item(gb_indices[1])

            data = torch.concat([data, green_image, blue_image], dim=0)
            label = label + 10 * green_label + 100 * blue_label
            target = torch.cat([target, green_target, blue_target], dim=0).reshape(3, 10)
    
        return data, target, label
    
    def __len__(self) -> int:
        return len(self.data.data)
    
    def _get_random_item(self, index):
        data = self.transforms(self.data.data[[index]])
        label = self.target_transforms(self.data.targets[index])
        target = OneHot()(label)
        
        if (DataMode.MISSING in self.mode) and label == 8:
            new_index = np.random.randint(len(self.data.data))
            data, target, label = self._get_random_item(new_index)

        return data, target, label