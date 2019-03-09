import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math

class LinearSequential(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin
            cos(theta + m)
        """
    def __init__(self, in_features, layers=[128, 2], dropout_probs=[0.5, 0.]):
        super(LinearSequential, self).__init__()
        self.in_features = in_features
        self.sequential = self.__make_layers__(layers, dropout_probs)

    def __make_layers__(self, layer_sizes, dropout_probs):
        layers = []
        input_size = self.in_features
        for layer_size, dropout_p in zip(layer_sizes, dropout_probs):
            output_size = layer_size
            layers.append(nn.Linear(input_size, output_size))
            layers.append(nn.ReLU(inplace=True))
            if dropout_p > 0 and dropout_p < 1:
                layers.append(nn.Dropout(p=dropout_p))
            input_size = output_size

        return nn.Sequential(*layers)

    def forward(self, input, target=None):
        return self.sequential(input)

class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin
            cos(theta + m)
        """
    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=True):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label=None):
        if self.training:
            # --------------------------- cos(theta) & phi(theta) ---------------------------
            cosine = F.linear(F.normalize(input), F.normalize(self.weight))
            sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
            phi = cosine * self.cos_m - sine * self.sin_m
            if self.easy_margin:
                phi = torch.where(cosine > 0, phi, cosine)
            else:
                phi = torch.where(cosine > self.th, phi, cosine - self.mm)
            # --------------------------- convert label to one-hot ---------------------------
            # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
            one_hot = torch.zeros(cosine.size(), device='cuda')
            one_hot.scatter_(1, label.view(-1, 1).long(), 1)
            # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
            output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
            output *= self.s
        else:
            cosine = F.linear(F.normalize(input), F.normalize(self.weight))
            output = self.s * cosine

        return output

class ArcMarginProduct_v2(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin
            cos(theta + m)
        """
    def __init__(self, in_features, out_features, m=0.50, easy_margin=True):
        super(ArcMarginProduct_v2, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.m = m
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label=None, alpha=0.1):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        input_norm = input.norm(2, 1, True).clamp(min=1e-12)
        crossentropy_output = F.linear(input, self.weight)
        if self.training:
            # --------------------------- cos(theta) & phi(theta) ---------------------------
            sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
            phi = cosine * self.cos_m - sine * self.sin_m
            if self.easy_margin:
                phi = torch.where(cosine > 0, phi, cosine)
            else:
                phi = torch.where(cosine > self.th, phi, cosine - self.mm)
            # --------------------------- convert label to one-hot ---------------------------
            # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
            one_hot = torch.zeros(cosine.size(), device='cuda')
            one_hot.scatter_(1, label.view(-1, 1).long(), 1)
            # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
            arcmargin_output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        else:
            arcmargin_output = cosine

        arcmargin_output = arcmargin_output * input_norm.expand_as(arcmargin_output)
        output = alpha * arcmargin_output + (1 - alpha) * crossentropy_output
        return output

class ArcMarginProduct_v3(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin
            cos(theta + m)
        """
    def __init__(self, in_features, out_features, m=0.01, easy_margin=True):
        super(ArcMarginProduct_v3, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.m = m
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label=None, alpha=0.1):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        input_norm = input.norm(2, 1, True).clamp(min=1e-12)
        crossentropy_output = F.linear(input, self.weight)
        if self.training:
            # --------------------------- cos(theta) & phi(theta) ---------------------------
            sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
            phi = cosine * self.cos_m - sine * self.sin_m
            if self.easy_margin:
                phi = torch.where(cosine > 0, phi, cosine)
            else:
                phi = torch.where(cosine > self.th, phi, cosine - self.mm)
            # --------------------------- convert label to one-hot ---------------------------
            # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
            one_hot = torch.zeros(cosine.size(), device='cuda')
            # Only for positive class
            one_hot[:,1] = label
            # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
            arcmargin_output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        else:
            arcmargin_output = cosine

        arcmargin_output = arcmargin_output * input_norm.expand_as(arcmargin_output)
        output = alpha * arcmargin_output + (1 - alpha) * crossentropy_output
        return output