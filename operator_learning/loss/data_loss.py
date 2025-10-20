import torch
import torch.nn as nn
# Store all loss classes in a dictionary using the register decorator 
LOSSES_CLASSES = {}

def register(cls):
    assert hasattr(cls, '__call__') and callable(cls), "loss class must implement the __call__ method"
    LOSSES_CLASSES[cls.__name__] = cls
    return cls

@register
class LpLoss(object):
    """
    Model loss
    Args:
        d (int): dimension
        p (int): order of norm 
        size_average (bool): take average
        reduction (bool): perform reduction
    """
    def __init__(self, 
                 d:int=2,
                 p:int=2,
                 size_average:bool=True, 
                 reduction:bool=True,
                 device=None
    ):
        super().__init__()
        # Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average
        self.device = device

    def abs(self, x, y):
        num_examples = x.size()[0]

        # Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h**(self.d/self.p))*torch.norm(x.flatten(num_examples,-1) - y.flatten(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)

        return diff_norms/y_norms

    def __call__(self, pred, ref):
        return self.rel(pred, ref)

@register
class VectorNormLoss(object):
    """
    Vector norm model loss
    Args:
        p (int): order of norm 
        dim (int): spatial dimension
        out: model prediction of shape [nBatch,nVar,nX,nY,(nZ)]
        ref: data reference of shape [nBatch,nVar,nX,nY,(nZ)]
    """
    def __init__(self, 
                 p:int=2,
                 dim:int=2,
                 absolute=False,
                 device=None
    ):
        super().__init__()
        # Lp-norm type is positive
        assert p > 0
        self.p = p
        self.dim = dim
        self.device = device
        if absolute:
            self.__class__.__call__ = self.__class__.__call__ABS
        self.absolute = absolute

    def vectorNorm(self, x):
        return torch.linalg.vector_norm(x, ord=self.p, dim=tuple(range(-self.dim, 0)))
    
    def __call__(self, pred, ref):
        refNorms = self.vectorNorm(ref)
        diffNorms = self.vectorNorm(pred-ref)
        return torch.mean(diffNorms/refNorms)

    def __call__ABS(self, pred, ref):
        return torch.mean(self.vectorNorm(pred-ref))
    
# ------------------------------
# Torch built-in loss wrappers
# ------------------------------

@register
class L1Loss(nn.Module):
    """Wrapper around torch.nn.L1Loss"""
    def __init__(self, reduction: str = "mean", device=None):
        super().__init__()
        self.loss_fn = nn.L1Loss(reduction=reduction)

    def __call__(self, pred, ref):
        if pred.is_complex():
            loss = (
                self.loss_fn(pred.real.flatten(start_dim=1), ref.real.flatten(start_dim=1))
                + self.loss_fn(pred.imag.flatten(start_dim=1), ref.imag.flatten(start_dim=1))
            )
        else:
            loss = self.loss_fn(pred.flatten(start_dim=1), ref.flatten(start_dim=1))

        return loss
    
@register
class MSELoss(nn.Module):
    """Wrapper around torch.nn.MSELoss"""
    def __init__(self, reduction: str = "mean", device=None):
        super().__init__()
        self.loss_fn = nn.MSELoss(reduction=reduction)

    def __call__(self, pred, ref):
        if pred.is_complex():
            loss = (
                self.loss_fn(pred.real.flatten(start_dim=1), ref.real.flatten(start_dim=1))
                + self.loss_fn(pred.imag.flatten(start_dim=1), ref.imag.flatten(start_dim=1))
            )
        else:
            loss = self.loss_fn(pred.flatten(start_dim=1), ref.flatten(start_dim=1))

        return loss
