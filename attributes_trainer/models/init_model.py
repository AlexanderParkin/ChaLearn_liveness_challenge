from .architectures import resnet_caffe, ResNext

from .layers import ArcMarginProduct, ArcMarginProduct_v2, ArcMarginProduct_v3, LinearSequential
import torch
import torch.nn as nn


def get_model(net_type, loss_type, pretrained=False, output_size=2):
    if net_type == 'Resnet_Caffe34':
        model = resnet_caffe.resnet_caffe34(pretrained)
    elif net_type == 'ResNext50':
        model = ResNext.resnext50()
    else:
        raise Exception('Unknown architecture type')
    
    if loss_type == 'cce':
        classifier = nn.Linear(512, output_size)
    elif loss_type == 'cce_128':
        classifier = nn.Linear(128, output_size)
    elif loss_type == 'cce_sequential':
        classifier = LinearSequential(512, [128, 1], [0.2, 0.])
    elif loss_type == 'cce_arc_margin_v3_1e-2_alpha_1e-1':
        classifier = ArcMarginProduct_v3(512, output_size, m=0.01)
    elif loss_type == 'arc_margin':
        classifier = ArcMarginProduct(512, output_size)
    elif loss_type == 'bce':
        classifier = nn.Linear(512, 1)
    elif loss_type == 'mse':
        classifier = nn.Linear(512, 1)
    else:
        raise Exception('Unknown loss type')
    
    return model, classifier       
  
def init_loss(criterion_name):

    if criterion_name=='bce':
        loss = nn.BCEWithLogitsLoss()
    elif criterion_name.startswith('arc_margin'):
        loss = nn.CrossEntropyLoss()
    elif 'cce' in criterion_name:
        loss = nn.CrossEntropyLoss()
    elif criterion_name == 'mse':
        loss = nn.MSELoss(reduction='mean')
    else:
        raise Exception('This loss function is not implemented yet.') 

    return loss 
