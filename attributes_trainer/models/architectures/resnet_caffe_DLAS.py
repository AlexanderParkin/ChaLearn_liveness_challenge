import torch.nn as nn  # type: ignore
import torch  # type: ignore
import torch.nn.init  # type: ignore
import math  # type: ignore
from collections import OrderedDict
#import models

use_relu = False
use_bn = True


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    global use_bn
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=not use_bn)


def calculate_scale(data):
    if data.dim() == 2:
        scale = math.sqrt(3 / data.size(1))
    else:
        scale = math.sqrt(3 / (data.size(1) * data.size(2) * data.size(3)))
    return scale

class ChannelAttention(nn.Module):
    def __init__(self, input_size, bottleneck_size):
        super(ChannelAttention, self).__init__()
        self.input_size = input_size
        self.bottleneck_size = bottleneck_size

        self.se_fc1 = nn.Conv2d(self.input_size, self.bottleneck_size, kernel_size = 1)
        self.se_fc2 = nn.Conv2d(self.bottleneck_size, self.input_size, kernel_size = 1)

    def forward(self, x):
        w_max = nn.functional.max_pool2d(x, x.size(2))
        w_max = nn.functional.relu(self.se_fc1(w_max))

        w_avg = nn.functional.avg_pool2d(x, x.size(2))
        w_avg = nn.functional.relu(self.se_fc1(w_avg))

        w = w_max + w_avg
        w = torch.sigmoid(self.se_fc2(w))

        x = x * w
        return x

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size):
        super(SpatialAttention, self).__init__()
        self.kernel_size = kernel_size

        self.conv1 = nn.Conv2d(2, 1, kernel_size = self.kernel_size, padding=self.kernel_size//2)

    def forward(self, x):
        w_max, _ = torch.max(x, dim=1, keepdim=True)
        w_avg = torch.mean(x, dim=1, keepdim=True)

        w = torch.cat([w_avg, w_max], dim=1)
        w = self.conv1(w)
        w = torch.sigmoid(w)
        x = x * w
        return x

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        global use_bn
        self.use_bn = use_bn
        self.conv1 = conv3x3(inplanes, planes, stride)

        torch.nn.init.normal_(self.conv1.weight.data, 0, 0.01)
        if self.conv1.bias is not None:
            self.conv1.bias.data.zero_()
        if self.use_bn:
            self.bn1 = nn.BatchNorm2d(planes)
        if use_relu:
            self.relu1 = nn.ReLU(inplace=True)
        else:
            self.relu1 = nn.PReLU(planes)
        self.conv2 = conv3x3(planes, planes)

        torch.nn.init.normal_(self.conv2.weight.data, 0, 0.01)
        if self.conv2.bias is not None:
            self.conv2.bias.data.zero_()
        if self.use_bn:
            self.bn2 = nn.BatchNorm2d(planes)
        if use_relu:
            self.relu2 = nn.ReLU(inplace=True)
        else:
            self.relu2 = nn.PReLU(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        if self.use_bn:
            out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        if self.use_bn:
            out = self.bn2(out)
        out = self.relu2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual

        return out

class AggBlock(nn.Module):
    def __init__(self, layer_lvl, layers, agg_type='concat', channel_attention=False):
        super(AggBlock, self).__init__()
        self.layer_lvl = layer_lvl
        self.agg_type = agg_type
        self.channel_attention = channel_attention

        if self.layer_lvl == 1:
            inplanes = 64
            outplanes = 128
        elif self.layer_lvl == 2:
            inplanes = 128
            outplanes = 256
        elif self.layer_lvl == 3:
            inplanes = 256
            outplanes = 512

        self.agg_layer = _make_layer(BasicBlock, inplanes, outplanes, layers, stride=2)

        if self.channel_attention:
            self.ch_att = ChannelAttention(outplanes, outplanes//4)

        if self.agg_type == 'concat':
            self.conv1 = nn.Conv2d(inplanes * 3, inplanes, kernel_size=1)

    def forward(self, prev_x, rgb_x, depth_x, ir_x):
        if self.agg_type == 'concat':
            x = torch.cat((rgb_x,depth_x,ir_x), dim=1)
            x = nn.functional.relu(self.conv1(x))

        if self.layer_lvl in [2,3]:
            x = prev_x + x

        x = self.agg_layer(x)

        if self.channel_attention:
            x = self.ch_att(x)

        return x

    def _load_pretrained_weights(self, weights):
        pretrained_weights = self.state_dict()
        replace_k = 'layer{}'.format(self.layer_lvl + 1)
        for k,v in pretrained_weights.items():
            if 'num_batches_tracked' in k or k.startswith('conv1') or 'ch_att' in k:
                continue
            pretrained_weights[k] = weights[k.replace('agg_layer', replace_k)]
        self.load_state_dict(pretrained_weights)

def _make_layer(block, inplanes, planes, blocks, stride=1):
        layers = []
        layers.append(nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=0,
                                bias=False))

        scale = calculate_scale(layers[-1].weight.data)
        torch.nn.init.uniform_(layers[-1].weight.data, -scale, scale)

        if layers[-1].bias is not None:
            layers[-1].bias.data.zero_()
        
        layers.append(nn.BatchNorm2d(planes))
        
        layers.append(nn.PReLU(planes))
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0))

        inplanes = planes
        for i in range(0, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

class ResNetCaffe(nn.Module):

    def __init__(self, layers, 
                 block=None, 
                 k=1, 
                 use_bn_=True, 
                 init='kaiming_normal', 
                 channel_attention=False):
        global use_relu
        global use_bn
        self.use_bn = use_bn
        self.channel_attention = channel_attention

        super(ResNetCaffe, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=0, bias=False)

        scale = calculate_scale(self.conv1.weight.data)
        torch.nn.init.uniform_(self.conv1.weight.data, -scale, scale)

        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.PReLU(32)

        block = block if block is not None else BasicBlock

        self.layer1 = _make_layer(block, 32, 64, layers[0])
        self.layer2 = _make_layer(block, 64, 128, layers[1], stride=2)
        self.layer3 = _make_layer(block, 128, 256, layers[2], stride=2)

        if self.channel_attention:
            self.ch_att1 = ChannelAttention(64, 64//4)
            self.ch_att2 = ChannelAttention(128, 128//4)
            self.ch_att3 = ChannelAttention(256, 256//4)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)        

        x_layer1 = self.layer1(x)
        if self.channel_attention:
            x_layer1 = self.ch_att1(x_layer1)

        x_layer2 = self.layer2(x_layer1)
        if self.channel_attention:
            x_layer2 = self.ch_att2(x_layer2)

        x_layer3 = self.layer3(x_layer2)
        if self.channel_attention:
            x_layer3 = self.ch_att3(x_layer3)

        return x_layer1, x_layer2, x_layer3

    def _load_pretrained_weights(self, weights):
        pretrained_weights = self.state_dict()
        for k,v in pretrained_weights.items():
            if 'num_batches_tracked' in k or 'ch_att' in k:
                continue
            pretrained_weights[k] = weights[k]
        self.load_state_dict(pretrained_weights)

class ResNetDLAS(nn.Module):
    def __init__(self, block, layers, DLAS_type='A', pretrained=False):
        super(ResNetDLAS, self).__init__()
        global use_bn
        self.use_bn = use_bn
        self.DLAS_type = DLAS_type
        self.channel_attention = False

        if self.DLAS_type == 'A':
            self.layer4 = _make_layer(block, 256 * 3, 512, layers[3], stride=2)
            self.main_avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.agg_avgpool = nn.AdaptiveAvgPool2d((1,1))
            self.agg_layer1 = AggBlock(layer_lvl=1, layers=layers[1], agg_type='concat')
            self.agg_layer2 = AggBlock(layer_lvl=2, layers=layers[2], agg_type='concat')
            self.agg_layer3 = AggBlock(layer_lvl=3, layers=layers[3], agg_type='concat')
        elif self.DLAS_type == 'B':
            self.channel_attention = True
            self.layer4 = _make_layer(block, 256 * 3, 512, layers[3], stride=2)
            self.ch_att4 = ChannelAttention(512, 512//4)

            self.main_avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.agg_avgpool = nn.AdaptiveAvgPool2d((1,1))

            self.agg_layer1 = AggBlock(layer_lvl=1, layers=layers[1], 
                                       agg_type='concat', channel_attention=self.channel_attention)
            self.agg_layer2 = AggBlock(layer_lvl=2, layers=layers[2], 
                                       agg_type='concat', channel_attention=self.channel_attention)
            self.agg_layer3 = AggBlock(layer_lvl=3, layers=layers[3], 
                                       agg_type='concat', channel_attention=self.channel_attention)
        elif self.DLAS_type == 'C':
            self.layer4 = _make_layer(block, 256 * 3, 512, layers[3], stride=2)
            self.main_avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.agg_avgpool = nn.AdaptiveAvgPool2d((1,1))
            self.agg_layer1 = AggBlock(layer_lvl=1, layers=layers[1], agg_type='concat')
            self.agg_layer2 = AggBlock(layer_lvl=2, layers=layers[2], agg_type='concat')
            self.agg_layer3 = AggBlock(layer_lvl=3, layers=layers[3], agg_type='concat')

            self.main_bottleneck = nn.Conv2d(512, 128, kernel_size=1)
            self.agg_bottleneck = nn.Conv2d(512,128, kernel_size=1)

        self.rgb_backbone = ResNetCaffe(layers[:3], channel_attention=self.channel_attention)
        self.depth_backbone = ResNetCaffe(layers[:3], channel_attention=self.channel_attention)
        self.ir_backbone = ResNetCaffe(layers[:3], channel_attention=self.channel_attention)


        if pretrained:
            self._load_pretrained_weights()

    def forward(self, x, y, z):
        x_layer1, x_layer2, x_layer3 = self.rgb_backbone(x)
        y_layer1, y_layer2, y_layer3 = self.depth_backbone(y)
        z_layer1, z_layer2, z_layer3 = self.ir_backbone(z)

        x = torch.cat((x_layer3,y_layer3,z_layer3), dim=1)
        x = self.layer4(x)
        if self.channel_attention:
            x = self.ch_att4(x)

        x = self.main_avgpool(x)

        agg_layer1 = self.agg_layer1(None, x_layer1, y_layer1, z_layer1)
        agg_layer2 = self.agg_layer2(agg_layer1, x_layer2, y_layer2, z_layer2)
        agg_layer3 = self.agg_layer3(agg_layer2, x_layer3, y_layer3, z_layer3)
        agg_x = self.agg_avgpool(agg_layer3)

        if self.DLAS_type == 'C':
            x = self.main_bottleneck(x)
            agg_x = self.agg_bottleneck(agg_x)

        x = x + agg_x
        x = x.view(x.size(0), -1)

        return x

    def _load_pretrained_weights(self):
        weights = torch.load('/media2/a.parkin/codes/Liveness_challenge/models/pretrained/resnet_caffe_mcs_orgl.pth', map_location='cpu')
        self.rgb_backbone._load_pretrained_weights(weights)
        self.depth_backbone._load_pretrained_weights(weights)
        self.ir_backbone._load_pretrained_weights(weights)
        self.agg_layer1._load_pretrained_weights(weights)
        self.agg_layer2._load_pretrained_weights(weights)
        self.agg_layer3._load_pretrained_weights(weights)

        pretrained_weights = self.layer4.state_dict()
        for k,v in pretrained_weights.items():
            if 'num_batches_tracked' in k:
                continue
            weight = weights['layer4.' + k]
            if (self.DLAS_type in ['A', 'B', 'C']) and (k == '0.weight'):
                weight = torch.cat((weight, weight, weight), dim=1)
            pretrained_weights[k] = weight
        self.layer4.load_state_dict(pretrained_weights)


def resnetDLAS_A(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetDLAS(BasicBlock, [1, 2, 5, 3], DLAS_type='A', pretrained=pretrained)
    return model

def resnetDLAS_B(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetDLAS(BasicBlock, [1, 2, 5, 3], DLAS_type='B', pretrained=pretrained)
    return model

def resnetDLAS_C(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetDLAS(BasicBlock, [1, 2, 5, 3], DLAS_type='C', pretrained=pretrained)
    return model

def test():
    model = resnetDLAS_C(pretrained=True)
    rgb = torch.rand(2, 3, 112, 112)
    ir = torch.rand(2, 3, 112, 112)
    depth = torch.rand(2, 3, 112, 112)
    print(model(rgb, depth, ir).size())