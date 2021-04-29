import torch
import torch.nn as nn
from torch import autograd

class SwishImplementation(autograd.Function):
    @staticmethod
    def forward(ctx, i):
        ctx.save_for_backward(i)
        return i * torch.sigmoid(i)

    @staticmethod
    def backward(ctx, grad_output):
        sigmord_i = torch.sigmoid(ctx.saved_variables[0])
        return grad_output * (sigmord_i * (1 + ctx.saved_variables[0] * (1 - sigmord_i)))


class Swish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)
