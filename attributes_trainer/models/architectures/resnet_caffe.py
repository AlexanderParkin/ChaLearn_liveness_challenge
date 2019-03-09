import torch.nn as nn  # type: ignore
import torch  # type: ignore
import torch.nn.init  # type: ignore
import math  # type: ignore
#import models

use_relu = False
use_bn = True

class SEBlock(nn.Module):
    def __init__(self, input_size, bottleneck_size):
        super(SEBlock, self).__init__()
        self.input_size = input_size
        self.bottleneck_size = bottleneck_size

        self.se_fc1 = nn.Conv2d(self.input_size, self.bottleneck_size, kernel_size = 1)
        self.se_fc2 = nn.Conv2d(self.bottleneck_size, self.input_size, kernel_size = 1)

    def forward(self, x):
        w_avg = nn.functional.avg_pool2d(x, x.size(2))
        w_avg = nn.functional.relu(self.se_fc1(w_avg))
        w = torch.sigmoid(self.se_fc2(w_avg))
        x = x * w
        return x

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


class ResNetCaffe(nn.Module):

    def __init__(self, layers, block=None, k=1, use_relu_=False,
                 use_bn_=True, init='kaiming_normal', bn_fc_mode=1, split_output=False,
                 split_size=512, descriptor_size=512, pretrained=None):
        global use_relu
        use_relu = use_relu_
        global use_bn
        use_bn = use_bn_
        self.use_bn = use_bn
        self.split_output = split_output
        self.bn_fc_mode = bn_fc_mode
        self.inplanes = round(32 * k)
        super(ResNetCaffe, self).__init__()
        self.conv1 = nn.Conv2d(3, round(32 * k), kernel_size=3, stride=1, padding=0,
                               bias=not use_bn)

        scale = calculate_scale(self.conv1.weight.data)
        torch.nn.init.uniform_(self.conv1.weight.data, -scale, scale)
        if self.conv1.bias is not None:
            self.conv1.bias.data.zero_()
        if self.use_bn:
            self.bn1 = nn.BatchNorm2d(round(32 * k))
        if use_relu:
            self.relu = nn.ReLU(inplace=True)
        else:
            self.relu = nn.PReLU(round(32 * k))

        block = block if block is not None else BasicBlock
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, round(64 * k), layers[0])
        self.layer2 = self._make_layer(block, round(128 * k), layers[1], stride=2)
        self.layer3 = self._make_layer(block, round(256 * k), layers[2], stride=2)
        self.layer4 = self._make_layer(block, round(512 * k), layers[3], stride=2)

        se_inplanes = 256
        self.se_block = SEBlock(256, 256//16)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        #self.fc = nn.Linear(512, descriptor_size)

        if pretrained:
            self._load_pretrained_weight(pretrained)
        else:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        '''
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        '''
        layers = []
        layers.append(nn.Conv2d(self.inplanes, planes, kernel_size=3, stride=1, padding=0,
                                bias=not self.use_bn))

        scale = calculate_scale(layers[-1].weight.data)
        torch.nn.init.uniform_(layers[-1].weight.data, -scale, scale)
        if layers[-1].bias is not None:
            layers[-1].bias.data.zero_()
        if self.use_bn:
            layers.append(nn.BatchNorm2d(planes))
        if use_relu:
            layers.append(nn.ReLU(inplace=True))
        else:
            layers.append(nn.PReLU(planes))
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0))

        self.inplanes = planes
        for i in range(0, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _load_pretrained_weight(self, pretrained):
        weights = torch.load('/media2/a.parkin/codes/Liveness_challenge/models/pretrained/resnet_caffe_mcs_orgl.pth', map_location='cpu')
        pretrained_weights = self.state_dict()
        for k,v in pretrained_weights.items():
            if 'num_batches_tracked' in k or 'se_block' in k:
                continue
            pretrained_weights[k] = weights[k]
        self.load_state_dict(pretrained_weights)

    def forward(self, x):
        x = self.conv1(x)
        if self.use_bn:
            x = self.bn1(x)
        x = self.relu(x)        

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.se_block(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

def resnet_caffe34(pretrained=None):
    model = ResNetCaffe([1,2,5,3], pretrained=pretrained)
    return model



def test():
    model = resnet_caffe34()
    rgb = torch.rand(2, 3, 112, 112)
    print(model(rgb).size())