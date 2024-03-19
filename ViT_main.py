import einops

from torchsummary import summary

import torch
from torch import nn
import torchvision
import torch.optim as optim
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, RandomHorizontalFlip, RandomCrop, RandomRotation

device = 'cuda' if torch.cuda.is_available() else 'cpu'