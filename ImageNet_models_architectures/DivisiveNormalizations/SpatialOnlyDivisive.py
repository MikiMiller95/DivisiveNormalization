import torch
import numbers
import warnings
import math
#from pytorch_complex_tensor import ComplexTensor
from torch.nn.parameter import Parameter
from torch.nn import Module
from torch.nn import functional as F
from torch.nn.modules import utils
from torch.nn import grad
from torch import __init__
from torch.nn import __init__
import torchvision
import time
import scipy
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack as pack



class SpatialOnlyDivisiveNorm(Module):
    r"""

    """
    #__constants__ = ['size', 'alpha', 'beta', 'k']

    def __init__(self, xy_lamb=10.,alpha=1., beta=0.25, k=1.):
        super(LearnSpatOnlyLocalResponseNorm, self).__init__()
        self.xy_lamb = Parameter(torch.Tensor([xy_lamb]))
        self.alpha = Parameter(torch.Tensor([alpha]))
        self.beta = Parameter(torch.Tensor([beta]))
        self.k = Parameter(torch.Tensor([k]))

    def forward(self, input):
        return spatial_only_divisive_normalization(input,self.xy_lamb, self.alpha, self.beta,
                                     self.k)

    def extra_repr(self):
        return 'xy_lambda={xy_lamb},alpha={alpha}, beta={beta}, k={k}'.format(**self.__dict__['_parameters'])

def spatial_only_divisive_normalization(input, xy_lamb=4, alpha=5, beta=0.15, k=1.):
    # type: (Tensor, int, float, float, float) -> Tensor
    """Applies local response normalization in three dimensions across the two spatial dimensions
    input: tensor with shape [batch x features x x_dim x y_dim]
    See :class:`~torch.nn.LocalResponseNorm` for details.
    """
    dim = input.dim()
    divi = input.mul(input).unsqueeze(1)
    # Get the sizes; clone the input so we can fill it with zeros and become the kernel
    sizes = input.size()
    print('sizes',sizes)
    weits = input.clone().detach()
    neighbors=len(input[0,:,0,0])
    xy_neighbors=len(input[0,0,:,0])
    batch_len=len(input[:,0,0,0])

    divi =divi.view(sizes[0],  sizes[1],sizes[2], sizes[3])
    divi=divi.unsqueeze(0)
    # Keep this an even number for now
    xy_padding=xy_neighbors#//2
    if xy_neighbors%2==0:
    #if xy_padding %2==0:
        xy_neighbors=xy_neighbors+1
        divi=F.pad(divi,(0,1,0,1,0,0))
        #divi=F.pad(divi,(0,0,0,0,0,0))
        #xy_padding=xy_padding+1
    elif xy_neighbors%2==1:
    #elif xy_padding%2==1:
        xy_neighbors=xy_neighbors
        #xy_padding=xy_padding
    else:
        pass
    feat_padding=0 #int(neighbors)#//2
    if neighbors%2==0:
    #if feat_padding %2==0:
        neighbors=neighbors+1
        #divi=F.pad(divi,(0,0,0,0,0,0))
        divi=F.pad(divi,(0,0,0,0,0,1))
        #feat_padding=feat_padding+1
    elif neighbors%2==1:
    #elif feat_padding%2==1:
        neighbors=neighbors
        #feat_padding=feat_padding
    elif neighbors%2==0 and xy_neighbors%2==0:
        divi=F.pad(divi,(0,1,0,1,0,1))
    else:
        pass

    #print('Padding in XY is :',xy_padding)
    #print('Padding in Feats is :',feat_padding)
    weits = weits.new_zeros(([int(neighbors)]+[int(xy_neighbors)]+[int(xy_neighbors)]))
    dev=input.get_device()
    ## building the indices in depthwise
    #idxs = torch.abs(torch.arange(int(neighbors),device='cuda:%d'%dev)-int(neighbors)//2)
    #idxs=idxs.unsqueeze(0).expand((xy_neighbors),(xy_neighbors),(neighbors))
    #idxs=idxs.transpose(0,2)
    ## indices in x and y 
    x_idxs = torch.abs(torch.arange(xy_neighbors,device='cuda:%d'%dev)-(xy_neighbors)//2).unsqueeze(0).expand((xy_neighbors),(xy_neighbors))
    y_idxs = x_idxs.transpose(0,1)
    xy_idxs= torch.sqrt((x_idxs.mul(x_idxs).add(y_idxs.mul(y_idxs))).float())
    xy_idxs = xy_idxs.unsqueeze(0)#.expand((xy_neighbors),(xy_neighbors))
    #print('Shape of xy_idxs')
    #print(xy_idxs.shape)
    # throw them into the exponentials
    weits[neighbors//2,:,:] = torch.exp(-xy_idxs/xy_lamb)#.mul(torch.exp(-idxs/lamb))
    #weits[:,:,:] = torch.exp(-xy_idxs/xy_lamb).mul(torch.exp(-idxs/lamb))
    # creating single dimension at 1;corresponds to number of input channels;only 1 input channel
    # 3D convolution; weits has dims: Cx1xCx1x1 ; this means we have C filters for the C channels
    divi=F.pad(divi,(xy_padding//2,xy_padding//2,xy_padding//2,xy_padding//2,feat_padding//2,feat_padding//2),value=0)
    weits = F.pad(weits,(xy_padding//2,xy_padding//2,xy_padding//2,xy_padding//2,feat_padding//2,feat_padding//2),value=0)
    # The div is the input**2; it has dimensions B x 1 x C x W x H
    # anchor of g is 0,0,0 (flip g and wrap circular)
    size_y=weits.size(-1)
    size_x=weits.size(-2)
    size_feats=weits.size(-3)
    weits_new = torch.zeros((size_feats,size_x,size_y),device='cuda:%d'%dev)
    weits_center_y = weits.size(-1) // 2
    weits_center_x = weits.size(-2) // 2
    weits_center_feats = weits.size(-3) // 2
    weits_feats, weits_x, weits_y = torch.meshgrid(torch.arange(weits.size(-3),device='cuda:%d'%dev), torch.arange(weits.size(-2),device='cuda:%d'%dev),torch.arange(weits.size(-1),device='cuda:%d'%dev))
    weits_new_y = (weits_y.flip(2) - weits_center_y) % weits_new.size(2)
    weits_new_x = (weits_x.flip(1) - weits_center_x) % weits_new.size(1)
    weits_new_feats = (weits_feats - weits_center_feats) % weits_new.size(0)
    weits_new[weits_new_feats, weits_new_x,weits_new_y] = weits[weits_feats, weits_x,weits_y]
    weits=weits_new.unsqueeze(0).unsqueeze(0)
    weits=weits.repeat(1,batch_len,1,1,1)
    
    weits=torch.rfft(weits,onesided=False,normalized=False,signal_ndim=3)
    divi=torch.rfft(divi,onesided=False,normalized=False,signal_ndim=3)

    real_divi=torch.mul(divi[:,:,:,:,:,0],weits[:,:,:,:,:,0])-torch.mul(divi[:,:,:,:,:,1],weits[:,:,:,:,:,1])
    img_divi=torch.mul(divi[:,:,:,:,:,0],weits[:,:,:,:,:,1])+torch.mul(divi[:,:,:,:,:,1],weits[:,:,:,:,:,0])
    divi=torch.stack((real_divi,img_divi),dim=-1)
    divi=torch.irfft(divi,normalized=False,onesided=False,signal_ndim=3)#,signal_sizes=before_divi_shape[2:])
    
    divi=divi/(xy_lamb+.000001)/(xy_lamb+.000001) #lambs
    if divi[:,:,:-1,xy_padding//2:-(xy_padding//2+1),xy_padding//2:-(xy_padding//2+1)].size()[-1] != input.size()[-1]:
        #print('inside differs')
        #print('divi',divi.size(), 'input ',input.size())
        divi=divi[:,:,:-1,xy_padding//2:-(xy_padding//2),xy_padding//2:-(xy_padding//2)]
    else:
        divi=divi[:,:,:-1,xy_padding//2:-(xy_padding//2+1),xy_padding//2:-(xy_padding//2+1)]
    divi = divi.mul(alpha).add(k).pow(beta)
    divi=divi.squeeze()
    return input.mul(input) / (divi+.000001)



