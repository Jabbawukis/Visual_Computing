import torch
from torch import Tensor

from torchvision.utils import make_grid
import torchvision.transforms as T


toPIL = T.ToPILImage()

def get_batch(train_data, batch_size):
    imgs = [train_data[i][0] for i in range(batch_size)]
    labels = [train_data[i][1] for i in range(batch_size)]
    inps = torch.stack(imgs)
    
    return inps, Tensor(labels)

def plot_examples(train_data, net=None):
    inps, labels = get_batch(train_data, 8)

    print("input size", inps[0].shape)
    
    print("inputs")
    display(toPIL(make_grid(inps)))
    
    print("ground truths")
    display(labels)
    
    if net is not None:
        preds = net(inps.cuda())
        print("predictions")
        display(preds.argmax(dim=1))