from __future__ import division
import torch
import numbers
import warnings
import math
from torch.nn.parameter import Parameter
from torch.nn import Module
from torch.nn import functional as F
from torch._C import _infer_size, _add_docstr
from torch.nn import _reduction as _Reduction
from torch.nn.modules import utils
from torch.nn.modules.utils import _single, _pair, _triple, _list_with_default
from torch.nn import grad
#from torch.nn import _VF
from torch._jit_internal import boolean_dispatch, List
from torch import __init__
from torch.nn import __init__
from torch.nn.modules._functions import CrossMapLRN2d as _cross_map_lrn2d
from torch._C import *
from itertools import repeat
import torch.onnx.symbolic_helper as sym_help
from torch.onnx.symbolic_helper import parse_args
import torchvision

class DivisiveNorm(Module):
    r"""

    """

    def __init__(self,lamb=5.,alpha=1., beta=0.75, k=1.):
        super(DivisiveNorm, self).__init__()
        #self.neighbors = Parameter(torch.Tensor([neighbors]))
        #self.neighbors = neighbors
        self.lamb = Parameter(torch.Tensor([lamb]))
        self.alpha = Parameter(torch.Tensor([alpha])) 
        self.beta = Parameter(torch.Tensor([beta])) 
        self.k = Parameter(torch.Tensor([k]))

    def forward(self, input):
        return divisive_normalization(input,self.lamb,self.alpha, self.beta,
                                     self.k)

    def extra_repr(self):
        return 'lambda={lamb},alpha={alpha}, beta={beta}, k={k}'.format(**self.__dict__['_parameters'])


def divisive_normalization(input, lamb=5,  alpha=5, beta=0.75, k=1.):
    """
    """
    neighbors=int(torch.ceil(2*4*lamb).item())
    if neighbors%2==0:
        neighbors=neighbors+1
    else:
        pass
    dim = input.dim()
    if dim < 3:
        raise ValueError('Expected 3D or higher dimensionality \
                         input (got {} dimensions)'.format(dim))
    div = input.mul(input).unsqueeze(1)
    #hacky trick to try and keep everything on cuda
    sizes = input.size()
    weits = input.clone().detach() 
    weits = weits.new_zeros(([1]+[1]+[int(neighbors)]+[1]+[1]))
    #weits = weits.new_zeros(list([sizes[1]])+[1]+list(sizes[1:3],[1])) #+[1,1]) #+list(sizes[1:]))
    if dim == 3:
        div = F.pad(div, (0, 0, neighbors // 2,  neighbors - 1 // 2))
        div = torch._C._nn.avg_pool2d((div,  neighbors, 1), stride=1).squeeze(1)
    else:
        dev = input.get_device()
        # indexx is a 1D tensor that is a symmetric exponential distribution of some "radius" neighbors
        #idxs = torch.abs(torch.arange(neighbors,device='cuda:%d'%dev)-neighbors//2)
        idxs = torch.abs(torch.arange(neighbors)-neighbors//2)
        weits[0,0,:,0,0]=idxs
        weits= torch.exp(-weits/lamb)
        # creating single dimension at 1;corresponds to number of input channels;only 1 input channel
        # 3D convolution; weits has dims: Cx1xCx1x1 ; this means we have C filters for the C channels
        # The div is the input**2; it has dimensions B x 1 x C x W x H
        div=F.conv3d(div,weits,padding=((neighbors//2),0,0))
        div=div/lamb

    div = div.mul(alpha).add(1).mul(k).pow(beta)
    div=div.squeeze()
    return input.mul(input) / div