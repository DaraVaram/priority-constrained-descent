"""The four CIFAR architectures of the paper (32x32 inputs, no max-pool after the stem).

Module names match the research code, so its checkpoints load with ``load_state_dict``. Each model's
``penalized_weights()`` returns the conv weights the secondary objectives act on, exactly as in the
paper's runs: every Conv2d except ResNet-34's three 1x1 projection shortcuts, DenseNet-121's stem and
MobileNetV2's depthwise convs. To run PCD on another architecture, give it the same method.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _init(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)


def _convs(model, skip=lambda name, conv: False):
    return [m.weight for n, m in model.named_modules() if isinstance(m, nn.Conv2d) and not skip(n, m)]


# ---------------------------------------------------------------------------------------- ResNet-34
class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, stride, bias=False), nn.BatchNorm2d(out_ch))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class ResNet34(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._stage(64, 64, 3, 1)
        self.layer2 = self._stage(64, 128, 4, 2)
        self.layer3 = self._stage(128, 256, 6, 2)
        self.layer4 = self._stage(256, 512, 3, 2)
        self.fc = nn.Linear(512, num_classes)
        _init(self)

    @staticmethod
    def _stage(in_ch, out_ch, blocks, stride):
        return nn.Sequential(BasicBlock(in_ch, out_ch, stride), *[BasicBlock(out_ch, out_ch) for _ in range(1, blocks)])

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer4(self.layer3(self.layer2(self.layer1(out))))
        return self.fc(F.adaptive_avg_pool2d(out, 1).flatten(1))

    def penalized_weights(self):
        return _convs(self, skip=lambda name, conv: "shortcut" in name)


# ------------------------------------------------------------------------------------- DenseNet-121
class DenseLayer(nn.Module):
    def __init__(self, in_ch, growth, bn_size):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.conv1 = nn.Conv2d(in_ch, bn_size * growth, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(bn_size * growth)
        self.conv2 = nn.Conv2d(bn_size * growth, growth, 3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        return torch.cat([x, out], dim=1)


class DenseBlock(nn.Module):
    def __init__(self, num_layers, in_ch, growth, bn_size):
        super().__init__()
        self.layers = nn.ModuleList(DenseLayer(in_ch + i * growth, growth, bn_size) for i in range(num_layers))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class Transition(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_ch)
        self.conv = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.pool = nn.AvgPool2d(2)

    def forward(self, x):
        return self.pool(self.conv(F.relu(self.bn(x))))


class DenseNet121(nn.Module):
    def __init__(self, num_classes=10, growth=32, block_layers=(6, 12, 24, 16), bn_size=4, compression=0.5):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        blocks, trans, ch = [], [], 64
        for i, n in enumerate(block_layers):
            blocks.append(DenseBlock(n, ch, growth, bn_size))
            ch += n * growth
            if i < len(block_layers) - 1:
                trans.append(Transition(ch, int(ch * compression)))
                ch = int(ch * compression)
        self.blocks = nn.ModuleList(blocks)
        self.trans = nn.ModuleList(trans)
        self.final_bn = nn.BatchNorm2d(ch)
        self.classifier = nn.Linear(ch, num_classes)
        _init(self)

    def forward(self, x):
        x = self.stem(x)
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i < len(self.trans):
                x = self.trans[i](x)
        x = F.relu(self.final_bn(x))
        return self.classifier(F.adaptive_avg_pool2d(x, 1).flatten(1))

    def penalized_weights(self):
        return _convs(self, skip=lambda name, conv: name.startswith("stem"))


# ------------------------------------------------------------------------------------ Inception-style
class ConvBnRelu(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class InceptionBlock(nn.Module):
    """Four parallel branches of ``branch_ch`` channels each: 1x1, 1x1->3x3, 1x1->5x5, maxpool->1x1."""

    def __init__(self, in_ch, branch_ch, stride=1):
        super().__init__()
        reduce_ch = max(branch_ch // 2, 16)
        self.branch1 = ConvBnRelu(in_ch, branch_ch, 1, stride=stride)
        self.branch2 = nn.Sequential(ConvBnRelu(in_ch, reduce_ch, 1),
                                     ConvBnRelu(reduce_ch, branch_ch, 3, stride=stride, padding=1))
        self.branch3 = nn.Sequential(ConvBnRelu(in_ch, reduce_ch, 1, stride=stride),
                                     ConvBnRelu(reduce_ch, branch_ch, 5, stride=1, padding=2))
        self.branch4_pool = nn.MaxPool2d(3, stride=stride, padding=1)
        self.branch4_conv = ConvBnRelu(in_ch, branch_ch, 1)

    def forward(self, x):
        return torch.cat([self.branch1(x), self.branch2(x), self.branch3(x),
                          self.branch4_conv(self.branch4_pool(x))], dim=1)


class Inception(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.stem = ConvBnRelu(3, 64, 3, padding=1)
        self.block1 = InceptionBlock(64, 64, stride=1)      # -> 256 channels
        self.block2 = InceptionBlock(256, 128, stride=2)    # -> 512 channels
        self.block3 = InceptionBlock(512, 192, stride=2)    # -> 768 channels
        self.classifier = nn.Linear(768, num_classes)
        _init(self)

    def forward(self, x):
        x = self.block3(self.block2(self.block1(self.stem(x))))
        return self.classifier(F.adaptive_avg_pool2d(x, 1).flatten(1))

    def penalized_weights(self):
        return _convs(self)


# -------------------------------------------------------------------------------------- MobileNetV2
class InvertedResidual(nn.Module):
    def __init__(self, in_ch, out_ch, stride, expand_ratio):
        super().__init__()
        hidden = in_ch * expand_ratio
        self.use_skip = stride == 1 and in_ch == out_ch
        layers = []
        if expand_ratio != 1:
            layers += [nn.Conv2d(in_ch, hidden, 1, bias=False), nn.BatchNorm2d(hidden), nn.ReLU6(inplace=True)]
        layers += [nn.Conv2d(hidden, hidden, 3, stride=stride, padding=1, groups=hidden, bias=False),
                   nn.BatchNorm2d(hidden), nn.ReLU6(inplace=True),
                   nn.Conv2d(hidden, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch)]
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.conv(x) if self.use_skip else self.conv(x)


class MobileNetV2(nn.Module):
    # (out_channels, blocks, stride, expand_ratio); CIFAR strides
    CONFIG = [(16, 1, 1, 1), (24, 2, 1, 6), (32, 3, 2, 6), (64, 4, 2, 6), (96, 3, 1, 6), (160, 3, 2, 6), (320, 1, 1, 6)]

    def __init__(self, num_classes=10):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU6(inplace=True))
        layers, ch = [], 32
        for out_ch, n, stride, t in self.CONFIG:
            for i in range(n):
                layers.append(InvertedResidual(ch, out_ch, stride if i == 0 else 1, t))
                ch = out_ch
        self.features = nn.Sequential(*layers)
        self.head_conv = nn.Sequential(nn.Conv2d(320, 1280, 1, bias=False), nn.BatchNorm2d(1280), nn.ReLU6(inplace=True))
        self.classifier = nn.Linear(1280, num_classes)
        _init(self)

    def forward(self, x):
        x = self.head_conv(self.features(self.stem(x)))
        return self.classifier(F.adaptive_avg_pool2d(x, 1).flatten(1))

    def penalized_weights(self):
        return _convs(self, skip=lambda name, conv: conv.groups > 1)


ARCHITECTURES = {"resnet34": ResNet34, "densenet121": DenseNet121, "inception": Inception, "mobilenetv2": MobileNetV2}


def build_model(arch: str, num_classes: int) -> nn.Module:
    return ARCHITECTURES[arch](num_classes=num_classes)
