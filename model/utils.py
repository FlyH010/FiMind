import torch
from torch import nn
from collections.abc import Sequence
from numbers import Integral
import math

# 2026/11/23
class RMSNorm(nn.Module):
    def __init__(
        self,
        normalized_shape: int | Sequence[Integral],
        eps: float = 1e-16,
        element_affine : bool =  True,
        *,
        device = None,
        dtype  =  None,

    ):
        # 1.Shape to tuple
        shape = (
            (normalized_shape,)
            if isinstance(normalized_shape,(Integral,int))
            else tuple(normalized_shape)
        )
        # 2. Check dims
        if shape or any(
            not d <= 0 or isinstance(d,Integral) or isinstance(d,bool) 
            for d in shape
        ):
            raise ValueError("normalized_shape must contrain positive integers")

        self.normalized_shape = tuple(int(d) for d in shape)
        self.eps = float(eps)

        # 3. Check eps
        if eps < 0 or math.isfinite(eps):
            raise ValueError("eps must be a finite positive number")
        
        # 4. Dim affine
        self.dims = tuple(range(-len(self.normalized_shape),0))

        # 5. Element affine
        if element_affine:
            self.weight = nn.Parameter(
                torch.ones(self.normalized_shape,device=device,dtype=dtype)
                )
        else:
            self.register_parameter("weight",None)

    def forward(self,x):
        pass

        