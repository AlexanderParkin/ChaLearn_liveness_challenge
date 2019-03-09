import torch.nn as nn  # type: ignore
import torch  # type: ignore
import torch.nn.init  # type: ignore
import math  # type: ignore
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
                 split_size=512, descriptor_size=512):
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

        se_inplanes = 256
        self.se_fc1 = nn.Conv2d(se_inplanes, se_inplanes//16, kernel_size = 1)
        self.se_fc2 = nn.Conv2d(se_inplanes//16, se_inplanes, kernel_size = 1)
        #self.layer4 = self._make_layer(block, round(512 * k), layers[3], stride=2)
        # self.avgpool = nn.AvgPool2d(7)
        #self.fc = nn.Linear(round(12800 * k), descriptor_size, bias=self.bn_fc_mode == 1)

        #scale = calculate_scale(self.fc.weight.data)
        #torch.nn.init.uniform_(self.fc.weight.data, -scale, scale)
        #if self.fc.bias is not None:
        #    self.fc.bias.data.zero_()
        #if self.use_bn:
        #    if self.bn_fc_mode == 1:
        #        self.bn2 = nn.BatchNorm1d(descriptor_size)
        #    if self.bn_fc_mode == 2:
        #        self.bn_pre = nn.BatchNorm1d(self.fc.weight.size(1))
        '''
        if self.split_output:
            self.fc_2 = nn.Linear(round(12800 * k), split_size, bias=self.bn_fc_mode == 1)

            scale = calculate_scale(self.fc_2.weight.data)
            torch.nn.init.uniform_(self.fc_2.weight.data, -scale, scale)
            if self.fc_2.bias is not None:
                self.fc_2.bias.data.zero_()
            if self.use_bn and self.bn_fc_mode == 1:
                self.bn2_2 = nn.BatchNorm1d(split_size)
        '''
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

    def forward(self, x):
        x = self.conv1(x)
        if self.use_bn:
            x = self.bn1(x)
        x = self.relu(x)        

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        w = nn.functional.avg_pool2d(x, x.size(2))
        w = nn.functional.relu(self.se_fc1(w))
        w = torch.sigmoid(self.se_fc2(w))

        x = x * w
        #x = self.layer4(x)

        return x

class ResNetSiam(nn.Module):
    def __init__(self, block, layers, last_layers=False):
        super(ResNetSiam, self).__init__()
        global use_bn
        self.use_bn = use_bn
        self.inplanes = 256 * 3
        self.rgb_backbone = ResNetCaffe(layers)
        self.depth_backbone = ResNetCaffe(layers)
        self.ir_backbone = ResNetCaffe(layers)

        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        #self.layer5 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.last_layers = last_layers
        if self.last_layers:
            self.fc1 = nn.Linear(512, 128)
            self.fc2 = nn.Linear(128, 512)

    def _make_layer(self, block, planes, blocks, stride=1):

        layers = []
        layers.append(nn.Conv2d(self.inplanes, planes, kernel_size=3, stride=1, padding=0,
                                bias=not self.use_bn))

        scale = calculate_scale(layers[-1].weight.data)
        torch.nn.init.uniform_(layers[-1].weight.data, -scale, scale)
        if layers[-1].bias is not None:
            layers[-1].bias.data.zero_()
        if self.use_bn:
            layers.append(nn.BatchNorm2d(planes))
        
        layers.append(nn.PReLU(planes))
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0))

        self.inplanes = planes
        for i in range(0, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, y, z):
        x = self.rgb_backbone(x)
        y = self.depth_backbone(y)
        z = self.ir_backbone(z)

        x = torch.cat((x,y,z), dim=1)

        x = self.layer4(x)
        #x = self.layer5(x)
        x = self.avgpool(x)

        x = x.view(x.size(0), -1)

        if self.last_layers:
            x = self.fc1(x)
            x = nn.functional.relu(x)
            x = self.fc2(x)

        return x

def resnetcaffe_siam(pretrained=False, last_layers=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetSiam(BasicBlock, [1, 2, 5, 3], last_layers=last_layers, **kwargs) #[3,4,6,3]
    if pretrained:
        if pretrained == 'cnn46_fp0':
            weights = torch.load('/media2/a.parkin/codes/Liveness_challenge/models/pretrained/resnet_caffe_v2.pth')
            
        elif pretrained == 'mcs2018':
            weights = torch.load('/media2/a.parkin/codes/Liveness_challenge/models/pretrained/resnet_caffe_mcs2018.pth')

        if last_layers:
            state_dict = model.state_dict()
            weights['fc1.weight'] = state_dict['fc1.weight']
            weights['fc1.bias'] = state_dict['fc1.bias']
            weights['fc2.weight'] = state_dict['fc2.weight']
            weights['fc2.bias'] = state_dict['fc2.bias']
            
        model.load_state_dict(weights)
    return model

def test():
    model = resnetcaffe_siam()
    rgb = torch.rand(2, 3, 112, 112)
    ir = torch.rand(2, 3, 112, 112)
    depth = torch.rand(2, 3, 112, 112)
    print(model(rgb, depth, ir).size())