import torch
import math

# Suppose the input image size is 224x224 with 3 color channels (RGB)
input_channels = 3

# Suppose we want to classify images into 10 different classes
num_classes = 10

FedNH_head_init = 'orthogonal'
dim = 2

# Example initialization based on FedNH_head_init setting
if FedNH_head_init == 'orthogonal':
    # Initializing with orthogonal initialization
    m, n = num_classes, input_channels
    prototype = torch.nn.init.orthogonal_(torch.rand(m, n))
    print(prototype)

elif FedNH_head_init == 'uniform' and dim == 2:
    # Uniform initialization in 2D space
    r = 1.0
    W = torch.zeros(num_classes, 2)
    for i in range(num_classes):
        theta = i * 2 * math.pi / num_classes
        W[i, :] = torch.tensor([r * math.cos(theta), r * math.sin(theta)])
    prototype = W
    print(prototype)
