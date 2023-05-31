import torch
from torch import nn

def get_lin(lin: nn.Linear, directions, direction_weights):
    with torch.no_grad():
        groups = len(direction_weights)
        new_lin = nn.Linear(groups * lin.in_features, groups * lin.out_features, device=lin.weight.device)
        
        new_ws = torch.zeros_like(new_lin.weight)
        new_bs = torch.zeros_like(new_lin.bias)
        
        for i, dw in enumerate(direction_weights):
            linw, linb = lin.weight.clone(), lin.bias.clone()
            for d, dwi in zip(directions, dw):
                linw += dwi * d[0]
                linb += dwi * d[1]
            new_ws[
                i*lin.out_features : (i+1)*lin.out_features,
                i*lin.in_features : (i+1)*lin.in_features] += linw
            new_bs[i*lin.out_features:(i+1)*lin.out_features] += linb
        
        new_lin.weight[:] = new_ws
        new_lin.bias[:] = new_bs

    return new_lin

def get_conv(conv: nn.Conv2d, directions, direction_weights):
    with torch.no_grad():
        groups = len(direction_weights)
        new_conv = nn.Conv2d(groups * conv.in_channels, groups * conv.out_channels, kernel_size=conv.kernel_size, groups=groups, device=conv.weight.device)
        
        new_conv_w = []
        new_conv_b = []
        
        for dw in direction_weights:
            c2w, c2b = conv.weight.clone(), conv.bias.clone()
            for d, dwi in zip(directions, dw):
                c2w += dwi * d[0]
                c2b += dwi * d[1]
            new_conv_w.append(c2w)
            new_conv_b.append(c2b)
        
        new_conv.weight[:] = torch.concat(new_conv_w)
        new_conv.bias[:] = torch.concat(new_conv_b)

    return new_conv