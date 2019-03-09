import torch
import torch.nn as nn
from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, PReLU, ReLU, Sigmoid, Dropout, MaxPool2d, \
    AdaptiveAvgPool2d, Sequential, Module
from collections import namedtuple


# Support: ['IR_50', 'IR_101', 'IR_152', 'IR_SE_50', 'IR_SE_101', 'IR_SE_152']

class Flatten(Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)

    return output


class SEModule(Module):
    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = AdaptiveAvgPool2d(1)
        self.fc1 = Conv2d(
            channels, channels // reduction, kernel_size=1, padding=0, bias=False)

        nn.init.xavier_uniform_(self.fc1.weight.data)

        self.relu = ReLU(inplace=True)
        self.fc2 = Conv2d(
            channels // reduction, channels, kernel_size=1, padding=0, bias=False)

        self.sigmoid = Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)

        return module_input * x


class bottleneck_IR(Module):
    def __init__(self, in_channel, depth, stride):
        super(bottleneck_IR, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(
                Conv2d(in_channel, depth, (1, 1), stride, bias=False), BatchNorm2d(depth))
        self.res_layer = Sequential(
            BatchNorm2d(in_channel),
            Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False), PReLU(depth),
            Conv2d(depth, depth, (3, 3), stride, 1, bias=False), BatchNorm2d(depth))

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)

        return res + shortcut


class bottleneck_IR_SE(Module):
    def __init__(self, in_channel, depth, stride):
        super(bottleneck_IR_SE, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(
                Conv2d(in_channel, depth, (1, 1), stride, bias=False),
                BatchNorm2d(depth))
        self.res_layer = Sequential(
            BatchNorm2d(in_channel),
            Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False),
            PReLU(depth),
            Conv2d(depth, depth, (3, 3), stride, 1, bias=False),
            BatchNorm2d(depth),
            SEModule(depth, 16)
        )

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)

        return res + shortcut


class Bottleneck(namedtuple('Block', ['in_channel', 'depth', 'stride'])):
    '''A named tuple describing a ResNet block.'''


def get_block(in_channel, depth, num_units, stride=2):

    return [Bottleneck(in_channel, depth, stride)] + [Bottleneck(depth, depth, 1) for i in range(num_units - 1)]


def get_blocks(num_layers):
    if num_layers == 50:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=4),
            get_block(in_channel=128, depth=256, num_units=14),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 100:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=13),
            get_block(in_channel=128, depth=256, num_units=30),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 152:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=8),
            get_block(in_channel=128, depth=256, num_units=36),
            get_block(in_channel=256, depth=512, num_units=3)
        ]

    return blocks

def _make_layer(block, unit_module):
    modules = []
    for bottleneck in block:
        modules.append(
            unit_module(bottleneck.in_channel,
                        bottleneck.depth,
                        bottleneck.stride))
    layer = Sequential(*modules)
    return layer

class AggBlock(nn.Module):
    def __init__(self, layer_lvl, block, unit_module, agg_type='concat'):
        super(AggBlock, self).__init__()
        self.layer_lvl = layer_lvl
        self.agg_type = agg_type

        self.agg_layer = _make_layer(block, unit_module)

        if self.agg_type == 'concat':
            inplanes = block[0].in_channel
            self.conv1 = nn.Conv2d(inplanes * 3, inplanes, kernel_size=1)

    def forward(self, prev_x, rgb_x, depth_x, ir_x):
        if self.agg_type == 'concat':
            x = torch.cat((rgb_x,depth_x,ir_x), dim=1)
            x = nn.functional.relu(self.conv1(x))

        if self.layer_lvl in [2,3]:
            x = prev_x + x

        x = self.agg_layer(x)
        return x

    def _load_pretrained_weights(self, weights):
        pretrained_weights = self.state_dict()
        for k,v in pretrained_weights.items():
            if k.startswith('conv1') or 'ch_att' in k:
                continue
            w_number = int(k.split('.')[1])
            if self.layer_lvl == 1:
                w_k = w_number + 3
            elif self.layer_lvl == 2:
                w_k = w_number + 7
            elif self.layer_lvl == 3:
                w_k = w_number + 21
            pretrained_weights[k] = weights[k.replace(f'agg_layer.{w_number}.', f'body.{w_k}.')]
        self.load_state_dict(pretrained_weights)

class Backbone(Module):
    def __init__(self, blocks, mode='ir'):
        super(Backbone, self).__init__()

        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        
        self.input_layer = Sequential(Conv2d(3, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))

        self.layer1 = _make_layer(blocks[0], unit_module)
        self.layer2 = _make_layer(blocks[1], unit_module)
        self.layer3 = _make_layer(blocks[2], unit_module)

        self._initialize_weights()

    def forward(self, x):
        x = self.input_layer(x)
        x_layer1 = self.layer1(x)
        x_layer2 = self.layer2(x_layer1)
        x_layer3 = self.layer3(x_layer2)

        return x_layer1, x_layer2, x_layer3

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def _load_pretrained_weights(self, weights):
        pretrained_weights = self.state_dict()
        for k,v in pretrained_weights.items():
            w_number = int(k.split('.')[1])
            if k.startswith('layer1.'):
                w_k = k.replace(f'layer1.{w_number}.', 'body.{0}.'.format(w_number))
            elif k.startswith('layer2.'):
                w_k = k.replace(f'layer2.{w_number}.', 'body.{0}.'.format(w_number + 3))
            elif k.startswith('layer3.'):
                w_k = k.replace(f'layer3.{w_number}.', 'body.{0}.'.format(w_number + 7))
            else:
                w_k = k
            pretrained_weights[k] = weights[w_k]
        self.load_state_dict(pretrained_weights)

class IR_DLAS(nn.Module):
    def __init__(self, input_size, num_layers, mode = 'ir', DLAS_type='A', pretrained=None):
        super(IR_DLAS, self).__init__()
        self.DLAS_type = DLAS_type

        assert input_size[0] in [112, 224], "input_size should be [112, 112] or [224, 224]"
        assert num_layers in [50, 100, 152], "num_layers should be 50, 100 or 152"
        assert mode in ['ir', 'ir_se'], "mode should be ir or ir_se"
        blocks = get_blocks(num_layers)

        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE

        '''
        if input_size[0] == 112:
            self.output_layer = Sequential(BatchNorm2d(512),
                                           Dropout(),
                                           Flatten(),
                                           Linear(512 * 7 * 7, 512))
        else:
            self.output_layer = Sequential(BatchNorm2d(512),
                                           Dropout(),
                                           Flatten(),
                                           Linear(512 * 14 * 14, 512))
        '''

        if self.DLAS_type == 'A':
            layer4_block = get_block(in_channel=256 * 3, depth=512, num_units=3)
            self.layer4 = _make_layer(layer4_block, unit_module)
            self.main_avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.agg_avgpool = nn.AdaptiveAvgPool2d((1,1))
            self.agg_layer1 = AggBlock(layer_lvl=1, block=blocks[1], unit_module=unit_module, agg_type='concat')
            self.agg_layer2 = AggBlock(layer_lvl=2, block=blocks[2], unit_module=unit_module, agg_type='concat')
            self.agg_layer3 = AggBlock(layer_lvl=3, block=blocks[3], unit_module=unit_module, agg_type='concat')

        self.rgb_backbone = Backbone(blocks[:3], mode=mode)
        self.depth_backbone = Backbone(blocks[:3], mode=mode)
        self.ir_backbone = Backbone(blocks[:3], mode=mode)

        if pretrained:
            self._load_pretrained_weights(pretrained)

    def forward(self, x, y, z):
        x_layer1, x_layer2, x_layer3 = self.rgb_backbone(x)
        y_layer1, y_layer2, y_layer3 = self.depth_backbone(y)
        z_layer1, z_layer2, z_layer3 = self.ir_backbone(z)

        x = torch.cat((x_layer3,y_layer3,z_layer3), dim=1)
        x = self.layer4(x)
        x = self.main_avgpool(x)

        agg_layer1 = self.agg_layer1(None, x_layer1, y_layer1, z_layer1)
        agg_layer2 = self.agg_layer2(agg_layer1, x_layer2, y_layer2, z_layer2)
        agg_layer3 = self.agg_layer3(agg_layer2, x_layer3, y_layer3, z_layer3)
        agg_x = self.agg_avgpool(agg_layer3)

        x = x + agg_x
        x = x.view(x.size(0), -1)

        return x

    def _load_pretrained_weights(self, pretrained):
        if pretrained == 'ms1m_epoch63':
            weights = torch.load('/media2/a.parkin/codes/Liveness_challenge/models/pretrained/backbone_ir50_ms1m_epoch63.pth', map_location='cpu')
        elif pretrained == 'ms1m_epoch120':
            weights = torch.load('/media2/a.parkin/codes/Liveness_challenge/models/pretrained/backbone_ir50_ms1m_epoch120.pth', map_location='cpu')
        elif pretrained == 'asia':
            weights = torch.load('/media2/a.parkin/codes/Liveness_challenge/models/pretrained/backbone_ir50_asia.pth', map_location='cpu')
        
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

            w_number = int(k.split('.')[0])
            w_k = w_number + 21
            weight = weights[k.replace(f'{w_number}.', f'body.{w_k}.', 1)]
            if (self.DLAS_type in ['A', 'B', 'C']) and (k.startswith('0.shortcut_layer.0')
                or k.startswith('0.res_layer.1')):
                weight = torch.cat((weight, weight, weight), dim=1)
            elif (self.DLAS_type in ['A', 'B', 'C']) and k.startswith('0.res_layer.0'):
                weight = torch.cat((weight, weight, weight), dim=0)
            pretrained_weights[k] = weight
        self.layer4.load_state_dict(pretrained_weights)
        '''
        pretrained_weights = self.output_layer.state_dict()
        for k,v in pretrained_weights.items():
            pretrained_weights[k] = weights['output_layer.' + k]
        self.output_layer.load_state_dict(pretrained_weights)
        '''

def IR_50(input_size):
    """Constructs a ir-50 model.
    """
    model = Backbone(input_size, 50, 'ir')

    return model

def IR_50_DLAS_A(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = IR_DLAS(input_size=(112,112), num_layers=50, mode='ir', DLAS_type='A', pretrained=pretrained)
    return model

def test():
    model = IR_50_DLAS_A()
    rgb = torch.rand(2, 3, 112, 112)
    ir = torch.rand(2, 3, 112, 112)
    depth = torch.rand(2, 3, 112, 112)
    print(model(rgb, depth, ir).size())