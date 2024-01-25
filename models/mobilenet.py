from itertools import chain

import timm
import torch.nn as nn

def get_max_depth(unfreeze_layers_depth_idx):
    max_depth = 0
    for depth_idx in unfreeze_layers_depth_idx:
        depth = depth_idx[0]
        if depth > max_depth:
            max_depth = depth
    return max_depth


def call_children_bfs(queue, unfreeze_layers_depth_idx, depth, max_depth):
    if depth > max_depth:
        return
    depth_idx_for_this_depth = [idx for idx in unfreeze_layers_depth_idx if idx[0] == depth]
    new_queue = []
    for i, child in enumerate(queue, start=1):
        if [depth, i] in depth_idx_for_this_depth:
            for param in child.parameters():
                param.requires_grad = True
        new_queue.append(child.children())
    new_queue = chain.from_iterable(new_queue)
    call_children_bfs(new_queue, unfreeze_layers_depth_idx, depth+1, max_depth)


def unfreeze_layer(model, unfreeze_layers_depth_idx):
    max_depth = get_max_depth(unfreeze_layers_depth_idx)
    queue = model.children()
    call_children_bfs(queue, unfreeze_layers_depth_idx, 1, max_depth)


class ApplyToSeq(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x):
        x = x.transpose(1, 2) # B, C, F, H, W > B, F, C, H, W
        b, f = x.shape[:2]
        y = x.reshape(b * f, *x.shape[2:])
        y = self.module(y)
        x = y.view(b, f, *y.shape[1:]) # B, F, C_out
        # x = x.transpose(1, 2) # B, F, C_out > B, C_out, F
        return x


class SelectFinalState(nn.Module):
    def forward(self, x):
        x = x[1]
        x = x.view(x.shape[1:])
        return x


def get_mobilenet(num_classes):
    feature_model = timm.create_model('tf_mobilenetv3_small_100.in1k', pretrained=True)

    unfreeze_layers_depth_idx = [
        [1, 5],
        [1, 8],
    ]

    for param in feature_model.parameters():
        param.requires_grad = False

    unfreeze_layer(feature_model, unfreeze_layers_depth_idx)

    model = nn.Sequential(
        ApplyToSeq(feature_model),
        nn.GRU(1000, 256, batch_first=True),
        SelectFinalState(),
        nn.Linear(256, 64),
        nn.Dropout(),
        nn.Linear(64, num_classes),
        nn.Softmax(dim=-1)
    )

    return model