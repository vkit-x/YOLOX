import os
import torch
from yolox.exp import Exp as MyExp
from yolox.data.datasets import COCODataset


class ExpL(MyExp):
    def __init__(self):
        super().__init__()
        self.depth = 1.0
        self.width = 1.0
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]


class ExpX(MyExp):
    def __init__(self):
        super().__init__()
        self.depth = 1.33
        self.width = 1.25
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]


def get_num_params(model: torch.nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def show():
    exp = ExpX()
    model = exp.get_model()
    num_backbone_params = get_num_params(model.backbone)
    # L: 46599040 (close to convnext small)
    # X: 87204000 (match convnext base)
    print('num_backbone_params', num_backbone_params)


def debug():
    exp = ExpL()
    model = exp.get_model()
    x = torch.rand((2, 3, 320, 320))
    # (Pdb) len(fs)
    # 3
    # (Pdb) fs[0].shape
    # torch.Size([1, 256, 40, 40])
    # (Pdb) fs[1].shape
    # torch.Size([1, 512, 20, 20])
    # (Pdb) fs[2].shape
    # torch.Size([1, 1024, 10, 10])
    # fs = model.backbone(x)
    targets = torch.as_tensor([[[0, 100.0, 100.0, 10.0, 10.0]], [[0, 100.0, 100.0, 10.0, 10.0]]])
    model(x, targets)
    breakpoint()


def debug_dataset():
    cc = COCODataset(json_file='instances_val2017.json', name='val2017')
    breakpoint()