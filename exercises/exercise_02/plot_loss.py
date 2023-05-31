import pickle
from tqdm.auto import tqdm
import plotly.graph_objects as go

import torch
from torch import nn

import numpy as np

from convert_net import get_conv, get_lin


def get_random_direction(states):
    random_direction = [torch.randn_like(w) for _, w in states.items()]

    for d, (_, w) in zip(random_direction, states.items()):
        if d.dim() <= 1:
            d.copy_(w) # keep directions for weights/bias that are only 1 per node
        else:
            for d_, w_ in zip(d, w):
                d_.mul_(w_.norm()/(d_.norm() + 1e-10))

    return random_direction

def get_loss_grid(net, data_loader, loss_fn, directions=None, resolution=(10,10), scale=1):
    with torch.no_grad():
        x, y = torch.linspace(-1*scale, 1*scale, resolution[0]), torch.linspace(-1*scale, 1*scale, resolution[1])
        Z = []

        for xi in tqdm(x):
            # transcribe net for parallel execution
            parallel_net = nn.Sequential()
            direction_weights = list(zip(xi.repeat(len(y)), y))
            dir_weight_id = 0
            for m in net:
                # deepcopy module
                m = pickle.loads(pickle.dumps(m))
                
                if isinstance(m, nn.Conv2d):
                    parallel_net.append(get_conv(m,
                                                 [
                                                    directions[0][dir_weight_id*2:(dir_weight_id + 1)*2],
                                                    directions[1][dir_weight_id*2:(dir_weight_id + 1)*2],
                                                 ], direction_weights))
                    dir_weight_id += 1
                elif isinstance(m, nn.Linear):
                    parallel_net.append(get_lin(m, [
                                                    directions[0][dir_weight_id*2:(dir_weight_id + 1)*2],
                                                    directions[1][dir_weight_id*2:(dir_weight_id + 1)*2],
                                                 ], direction_weights))
                    dir_weight_id += 1
                else:
                    parallel_net.append(m)
    
            parallel_net.eval().cuda()
            losses = []
            for inps, gts in data_loader:
                out = parallel_net(inps.cuda().repeat(1, resolution[1], 1, 1))
                
                for i in range(resolution[1]):
                    out = out.view(-1, resolution[1], net[-1].out_features)
                    losses.append(loss_fn(
                        out[:, i], gts.cuda()
                    ))
            l = torch.stack(losses).view(len(data_loader), -1)
            Z.append(l.mean(dim=0).cpu())

    return x, y, torch.stack(Z), directions

def get_state_directions(states, n_states=10, start_from=30, reference_id=-1):
    param_shapes = []
    A = []
    state_ids = np.linspace(start_from, len(states) - 1, n_states).round().astype(int)

    for param in states[0]:
        params = [s[param] for s in [states[i] for i in state_ids]]
        reference = params[reference_id]
        params = list(filter(lambda x: torch.any(x != reference), params))

        target_shape = params[0].shape
        param_shapes.append(target_shape)

        A.append(torch.stack(params).view(len(params), -1) - reference.flatten()[None])

    A = torch.concat(A, dim=1)

    _, _, V = torch.pca_lowrank(A)

    loss_coordinates = A.matmul(V[:, :2])

    reference = torch.zeros((1,2), device=loss_coordinates.device)
    if reference_id >= 0:
        loss_coordinates = torch.concat([
            loss_coordinates[:reference_id],
            reference,
            loss_coordinates[reference_id:]
        ], dim=0)
    elif reference_id == -1:
        loss_coordinates = torch.concat([loss_coordinates, reference], dim=0)
    else:
        loss_coordinates = torch.concat([
            loss_coordinates[:reference_id + 1],
            reference,
            loss_coordinates[reference_id + 1:]
        ], dim=0)

    loss_coordinates = loss_coordinates.T
    
    return [V[:, 0], V[:, 1]], state_ids, loss_coordinates

def plot_losses(losses):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.arange(len(losses)), y=losses))
    fig.update_xaxes(title_text='Step')
    fig.update_yaxes(title_text='Loss')
    return fig

def plot_contour(x, y, z, plot_size=(500, 400), scale=False):
    fig = go.Figure(data=go.Contour(
        x=x,
        y=y,
        z=z,
        contours=dict(
            coloring = 'lines',
            showlabels = not scale
        ),
        line_width = 2,
        showscale=scale),
        layout=dict(
            autosize = False,
            width = plot_size[0],
            height = plot_size[1],
            margin = dict(l=10,r=30,t=30,b=30)
        ))
    return fig

def plot_surface(x, y, z, plot_size=(600, 800)):
    fig = go.Figure(go.Surface(
            x=x,
            y=y,
            z=z,
            opacity=0.6
        ),
        layout=dict(
            autosize = False,
            width = plot_size[0],
            height = plot_size[1],
            margin = dict(l=10,r=10,t=30,b=30)
        ))
    fig.update_traces(contours_z=dict(show=True, usecolormap=True, project_z=True))
    return fig