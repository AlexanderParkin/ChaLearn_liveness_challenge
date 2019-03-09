from .architectures import MobileNetV2
from .architectures import resnet_siam
from .architectures import resnet_v2
from .architectures import ResNext
from .architectures import resnet_caffe
from .architectures import resnet_caffe_cbam
from .architectures import resnet_caffe_DLAS
from .architectures import ir_se_dlas
from .architectures import resnext_dlas

from .layers import ArcMarginProduct, ArcMarginProduct_v2, ArcMarginProduct_v3, LinearSequential
import torch
import torch.nn as nn


def get_model(net_type, clf_type, pretrained=False, output_size=2):
    if net_type == 'ResNet18Siam':
        model = resnet_siam.resnet18siam()
    elif net_type == 'ResNet34Siam':
        model = resnet_v2.resnet34siam(pretrained)
    elif net_type == 'ResNet152Siam':
        model = resnet_v2.resnet152siam(pretrained)
    elif net_type == 'ResNet34CaffeSiam':
        model = resnet_caffe.resnetcaffe_siam(pretrained)
    elif net_type == 'ResNet34CaffeSiam_fc':
        model = resnet_caffe.resnetcaffe_siam(pretrained, last_layers=True)
    elif net_type == 'ResNet34CaffeCBAMSiam':
        model = resnet_caffe_cbam.resnetcaffe_siam(pretrained)
    elif net_type == 'ResNext50Siam':
        model = ResNext.resnext50siam(pretrained)
    elif net_type == 'ResNet34Merge':
        model = resnet_v2.resnet34merge(pretrained)
    elif net_type == 'ResNet34DLAS_A':
        model = resnet_caffe_DLAS.resnetDLAS_A(pretrained)
    elif net_type == 'ResNet34DLAS_B':
        model = resnet_caffe_DLAS.resnetDLAS_B(pretrained)
    elif net_type == 'ResNet34DLAS_C':
        model = resnet_caffe_DLAS.resnetDLAS_C(pretrained)
    elif net_type == 'IR_50DLAS_A':
        model = ir_se_dlas.IR_50_DLAS_A(pretrained=pretrained)
    elif net_type == 'ResneXt50DLAS_A':
        model = resnext_dlas.resnext50DLAS(pretrained=pretrained)
    else:
        raise Exception('Unknown architecture type')
    

    if clf_type == 'linear':
        classifier = nn.Linear(512, output_size)
    elif clf_type == 'linear_sequential':
        classifier = LinearSequential(512, [128, 2], [0.5, 0.])
    elif clf_type == 'arc_margin':
        classifier = ArcMarginProduct(512, output_size)
    elif clf_type == 'arc_margin_5e-2':
        classifier = ArcMarginProduct(512, output_size, m=0.05)
    elif clf_type == 'cce_arc_margin_1e-1':
        classifier = ArcMarginProduct_v2(512, output_size, m=0.1)
    elif clf_type == 'cce_arc_margin_5e-2':
        classifier = ArcMarginProduct_v2(512, output_size, m=0.05)
    elif clf_type == 'cce_arc_margin_14e+1':
        classifier = ArcMarginProduct_v2(512, output_size, m=1.4)
    elif clf_type == 'cce_arc_margin_v3_1e-2_alpha_1e-1':
        classifier = ArcMarginProduct_v3(512, output_size, m=0.01)
    elif clf_type == 'linear_bce':
        classifier = nn.Linear(512, 1)
    else:
        raise Exception('Unknown clf type')
    
    return model, classifier   

class FocalLoss(nn.Module):
    def __init__(self, gamma = 2, eps = 1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.ce = nn.CrossEntropyLoss()

    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()
    
def init_loss(criterion_name):

    if criterion_name=='bce':
        loss = nn.BCEWithLogitsLoss()
    elif criterion_name=='cce':
        loss = nn.CrossEntropyLoss()
    elif criterion_name.startswith('arc_margin'):
        loss = nn.CrossEntropyLoss()
    elif 'cce' in criterion_name:
        loss = nn.CrossEntropyLoss()
    elif criterion_name == 'focal_loss':
        loss = FocalLoss()
    else:
        raise Exception('This loss function is not implemented yet.') 

    return loss 
