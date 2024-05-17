from torchvision import models
from torch import nn
import torch
import torch.functional as F
from .MetaModule import *
import math

def upsample_bilinear(x, size):
    return F.interpolate(x, size=size[2:], mode='bilinear', align_corners=False)

class Backbone(nn.Module):
    def __init__(self, pretrained=False):
        super(Backbone, self).__init__()

        model = list(models.vgg16(pretrained=pretrained).features.children())
        self.feblock1 = nn.Sequential(*model[:16])
        self.feblock2 = nn.Sequential(*model[16:23])
        self.feblock3 = nn.Sequential(*model[23:30])

        # backend
        self.beblock3 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.beblock2 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.beblock1 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.feblock1(x)
        x1 = x
        x = self.feblock2(x)
        x2 = x
        x = self.feblock3(x)

        # decoding stage
        x = self.beblock3(x)
        x3_ = x
        x = upsample_bilinear(x, x2.shape)
        x = torch.cat([x, x2], 1)

        x = self.beblock2(x)
        x2_ = x
        x = upsample_bilinear(x, x1.shape)
        x = torch.cat([x, x1], 1)

        x1_ = self.beblock1(x)

        x2_ = upsample_bilinear(x2_, x1.shape)
        x3_ = upsample_bilinear(x3_, x1.shape)

        x = torch.cat([x1_, x2_, x3_], 1)
        return x

class ChannelGate_sub(nn.Module):
    """A mini-network that generates channel-wise gates conditioned on input tensor."""

    def __init__(self, in_channels, num_gates=None, return_gates=False,
                 gate_activation='sigmoid', reduction=16, layer_norm=False):
        super(ChannelGate_sub, self).__init__()
        if num_gates is None:
            num_gates = in_channels
        self.return_gates = return_gates
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels//reduction, kernel_size=1, bias=True, padding=0)
        self.norm1 = None
        if layer_norm:
            self.norm1 = nn.LayerNorm((in_channels//reduction, 1, 1))
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_channels//reduction, num_gates, kernel_size=1, bias=True, padding=0)
        if gate_activation == 'sigmoid':
            self.gate_activation = nn.Sigmoid()
        elif gate_activation == 'relu':
            self.gate_activation = nn.ReLU(inplace=True)
        elif gate_activation == 'linear':
            self.gate_activation = None
        else:
            raise RuntimeError("Unknown gate activation: {}".format(gate_activation))

    def forward(self, x):
        input = x
        x = self.global_avgpool(x)
        x = self.fc1(x)
        if self.norm1 is not None:
            x = self.norm1(x)
        x = self.relu(x)
        x = self.fc2(x)
        if self.gate_activation is not None:
            x = self.gate_activation(x)
        if self.return_gates:
            return x
        return input * x, input * (1 - x), x


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class MetaMSNetBase(MetaModule):
    def __init__(self, pretrained=False):
        super(MetaMSNetBase, self).__init__()

        self.backbone = Backbone(True)

        self.output_layer = nn.Sequential(
            nn.Conv2d(256, 1, kernel_size=1, bias=False),
            nn.ReLU(inplace=True)
        )

        # self._initialize_weights()
        self.part_num = 1024
        variance = math.sqrt(1.0)
        self.sem_mem = nn.Parameter(torch.FloatTensor(1, 256, self.part_num).normal_(0.0, variance))
        self.sty_mem = nn.Parameter(torch.FloatTensor(4, 1, 256, self.part_num // 4).normal_(0.0, variance))
        self.sem_down = nn.Sequential(
            nn.Conv2d(512 + 256 + 128, 256, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)
        )
        self.sty_down = nn.Sequential(
            nn.Conv2d(512 + 256 + 128, 256, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)
        )

        # Instance normalization
        self.inplanes = 896 #
        self.conv = nn.Conv2d(64, 896, kernel_size=1)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.layer1 = self._make_layer(BasicBlock, 64, 2)

        # IN bridge:
        self.IN1 = nn.InstanceNorm2d(64, affine=True)

        # SE for selection:
        self.style_reid_laye1 = ChannelGate_sub(64, num_gates=64, return_gates=False,
                                                gate_activation='sigmoid', reduction=16, layer_norm=False)


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.InstanceNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def conv_features(self, x):
        x = self.backbone(x)
        feature = self.sty_down(x)

        return feature.unsqueeze(0)

    # same as network_forward
    def train_forward(self, x, label):

        size = x.shape

        x = self.backbone(x)
        x_style = x

        x_1 = self.layer1(x_style)
        x_IN_1 = self.IN1(x_1)
        x_style_1 = x_1 - x_IN_1
        x_style_1_reid_useful, x_style_1_reid_useless, selective_weight_useful_1 = self.style_reid_laye1(x_style_1)

        x_1 = x_style_1 + x_style_1_reid_useless

        x_style = self.conv(x_1)

        memory = self.sem_mem.repeat(x.shape[0], 1, 1)
        memory_key = memory.transpose(1, 2)  # bs*50*256
        sem_pre = self.sem_down(x)

        sty_style = self.sty_down(x_style)

        sty_pre = self.sty_down(x)

        sem_pre_ = sem_pre.view(sem_pre.shape[0], sem_pre.shape[1], -1)

        diLogits = torch.bmm(memory_key, sem_pre_)  # bs*50*(h*w)

        invariant_feature = torch.bmm(memory_key.transpose(1, 2), F.softmax(diLogits, dim=1))

        # calculate rec loss
        recon_sim = torch.bmm(invariant_feature.transpose(1, 2), sem_pre_)
        sim_gt = torch.linspace(0, sem_pre.shape[2] * sem_pre.shape[3] - 1,
                                sem_pre.shape[2] * sem_pre.shape[3]).unsqueeze(0).repeat(sem_pre.shape[0], 1).cuda()
        sim_loss = F.cross_entropy(recon_sim, sim_gt.long(), reduction='none') * 0.1

        invariant_feature_ = invariant_feature.view(invariant_feature.shape[0], invariant_feature.shape[1],
                                                   sem_pre.shape[2], sem_pre.shape[3])

        # density prediction
        den = self.output_layer(invariant_feature_)
        den = upsample_bilinear(den, size=size)

        # re-encoding ds features
        memory2 = self.sty_mem[label].cuda()

        memory2 = memory2.repeat(x_style.shape[0], 1, 1)

        mem2_key = memory2.transpose(1, 2)

        sty_style_ = sty_style.view(sty_style.shape[0], sty_style.shape[1], -1)

        dsLogits = torch.bmm(mem2_key, sty_style_)

        spe_feature = torch.bmm(mem2_key.transpose(1, 2), F.softmax(dsLogits, dim=1))
        # calculate rec loss with style features
        recon_sim2 = torch.bmm(spe_feature.transpose(1, 2), sty_style_)
        sim_gt2 = torch.linspace(0, sty_pre.shape[2] * sty_pre.shape[3] - 1,
                                 sty_pre.shape[2] * sty_pre.shape[3]).unsqueeze(0).repeat(sty_pre.shape[0], 1).cuda()
        sim_loss2 = F.cross_entropy(recon_sim2, sim_gt2.long(), reduction='sum') * 0.1
        # orthogonal loss between sem and sty features

        orth_pre = torch.bmm(sty_style_.transpose(1, 2), sem_pre_)

        orth_loss = 0.01 * torch.sum(torch.pow(torch.diagonal(orth_pre, dim1=-2, dim2=-1), 2))

        return den, sim_loss, sim_loss2, orth_loss


    def forward(self, x):
        size = x.shape

        x = self.backbone(x)

        # memory:
        memory = self.sem_mem.repeat(x.shape[0], 1, 1)
        memory_key = memory.transpose(1, 2)  # bs*50*256
        sem_pre = self.sem_down(x)

        sem_pre_ = sem_pre.view(sem_pre.shape[0], sem_pre.shape[1], -1)
        diLogits = torch.bmm(memory_key, sem_pre_)  # bs*50*(h*w)
        invariant_feature = torch.bmm(memory_key.transpose(1, 2), F.softmax(diLogits, dim=1))

        invariant_feature = invariant_feature.view(invariant_feature.shape[0], invariant_feature.shape[1],
                                                   sem_pre.shape[2], sem_pre.shape[3])

        den = self.output_layer(invariant_feature)

        den = upsample_bilinear(den, size=size)

        return den

class MetaMemNet(MetaModule):
    def getBase(self):
        baseModel = MetaMSNetBase(True)
        return baseModel

    def __init__(self):
        super(MetaMemNet, self).__init__()
        self.base = self.getBase()

    def train_forward(self, x, label):
        dens, sim_loss, sim_loss2, orth_loss = self.base.train_forward(x, label)

        dens = upsample_bilinear(dens, x.shape)

        return dens, sim_loss, sim_loss2, orth_loss

    def forward(self, x):
        dens = self.base(x)

        dens = upsample_bilinear(dens, x.shape)

        return dens

    def get_grads(self):
        grads = []
        for p in self.base.parameters():
            grads.append(p.grad.data.clone().flatten())
        return torch.cat(grads)

    def set_grads(self, new_grads):
        start = 0
        for k, p in enumerate(self.base.parameters()):
            dims = p.shape
            end = start + dims.numel()
            p.grad.data = new_grads[start:end].reshape(dims)
            start = end

    def conv_features(self, x):
        x = self.base.conv_features(x)

        return x
