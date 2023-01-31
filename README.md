# DivisiveNormalization
The relevant code for the 2022 ICLR paper:
Divisive Feature Normalization Improves Image Recognition Performance in AlexNet
by Michelle Miller, SueYeon Chung, Ken D Miller

Incorporating Divisive Normalization into AlexNet

This respository includes the architectures used for incorporating divisive normalization (DN) into 5 layer convolutional networks. 
We modeled DN as an exponential kernel of inhibition. This means, a neuron would be inhibited by it's neighbors according to this exponential weighting. The paper discusses this implementation along the feature dimensions. We introduce four learnable parameters which define the spatial scale of the kernel (lambda), the weightings of normalization (alpha), an overall weighting (k) and an exponent for the normalization (beta). 



Additional discussion:
Limited success was found in extending this to the spatial dimensions. 
