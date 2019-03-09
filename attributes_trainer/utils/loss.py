import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math

def ArcMarginOneClassLoss(descr, centriod, target, s=30.0, margin=0.50):
    batch_size = descr.size()[0]
    cos_m = math.cos(m)
    sin_m = math.sin(m)
    
    cosine = F.linear(F.normalize(descr,p=2, dim=2), 
                      F.normalize(centroid,p=2, dim=1))
    cosine = torch.where((target == 1).view(-1,1), cosine, -1.0 * cosine)
    sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
    phi = cosine * self.cos_m - sine * self.sin_m
    logit = torch.exp(phi) / (torch.exp(phi) + torch.exp(cosine))
    return -1.0 * torch.sum(torch.log(logit)) / batch_size