import torch
from torch import nn, Tensor
from torch.nn import functional as F


class BasicBlock(nn.Module):
    """2 Layer No Expansion Block
    """
    expansion: int = 1
    def __init__(self, c1, c2, s=1, downsample= None) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(c1, c2, 3, s, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(c2)
        self.conv2 = nn.Conv2d(c2, c2, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(c2)
        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None: identity = self.downsample(x)
        out += identity
        return F.relu(out)


class Bottleneck(nn.Module):
    """3 Layer 4x Expansion Block
    """
    expansion: int = 4
    def __init__(self, c1, c2, s=1, downsample=None) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(c1, c2, 1, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(c2)
        self.conv2 = nn.Conv2d(c2, c2, 3, s, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(c2)
        self.conv3 = nn.Conv2d(c2, c2 * self.expansion, 1, 1, 0, bias=False)
        self.bn3 = nn.BatchNorm2d(c2 * self.expansion)
        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None: identity = self.downsample(x)
        out += identity
        return F.relu(out)


resnet_settings = {
    '18': [BasicBlock, [2, 2, 2, 2]],
    '34': [BasicBlock, [3, 4, 6, 3]],
    '50': [Bottleneck, [3, 4, 6, 3]],
    '101': [Bottleneck, [3, 4, 23, 3]],
    '152': [Bottleneck, [3, 8, 36, 3]]
}


class ResNet(nn.Module):
    def __init__(self, model_name: str = '50', in_channels: int = 3) -> None:
        super().__init__()
        assert model_name in resnet_settings.keys(), f"ResNet model name should be in {list(resnet_settings.keys())}"
        block, depths = resnet_settings[model_name]

        self.inplanes = 64
        self.in_channels = in_channels
        self.conv1 = nn.Conv2d(in_channels, self.inplanes, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.maxpool = nn.MaxPool2d(3, 2, 1)

        self.layer1 = self._make_layer(block, 64, depths[0], s=1)
        self.layer2 = self._make_layer(block, 128, depths[1], s=2)
        self.layer3 = self._make_layer(block, 256, depths[2], s=2)
        self.layer4 = self._make_layer(block, 512, depths[3], s=2)


    def _make_layer(self, block, planes, depth, s=1) -> nn.Sequential:
        downsample = None
        if s != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, 1, s, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )
        layers = nn.Sequential(
            block(self.inplanes, planes, s, downsample),
            *[block(planes * block.expansion, planes) for _ in range(1, depth)]
        )
        self.inplanes = planes * block.expansion
        return layers


    def forward(self, x: Tensor) -> Tensor:
        x = self.maxpool(F.relu(self.bn1(self.conv1(x))))   # [1, 64, H/4, W/4]
        x1 = self.layer1(x)  # [1, 64/256, H/4, W/4]
        x2 = self.layer2(x1)  # [1, 128/512, H/8, W/8]
        x3 = self.layer3(x2)  # [1, 256/1024, H/16, W/16]
        x4 = self.layer4(x3)  # [1, 512/2048, H/32, W/32]
        return x1, x2, x3, x4

class ConvModule(nn.Sequential):
    def __init__(self, c1, c2, k, s=1, p=0):
        super().__init__(
            nn.Conv2d(c1, c2, k, s, p, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(True)
        )


class FPNHead(nn.Module):
    """Panoptic Feature Pyramid Networks
    https://arxiv.org/abs/1901.02446
    """
    def __init__(self, in_channels, channel=128, num_classes=19):
        super().__init__()
        self.lateral_convs = nn.ModuleList([])
        self.output_convs = nn.ModuleList([])

        for ch in in_channels[::-1]:
            self.lateral_convs.append(ConvModule(ch, channel, 1))
            self.output_convs.append(ConvModule(channel, channel, 3, 1, 1))

        self.conv_seg = nn.Conv2d(channel, num_classes, 1)
        self.dropout = nn.Dropout2d(0.1)

    def forward(self, features) -> Tensor:
        features = features[::-1]
        out = self.lateral_convs[0](features[0])

        for i in range(1, len(features)):
            out = F.interpolate(out, scale_factor=2.0, mode='nearest')
            out = out + self.lateral_convs[i](features[i])
            out = self.output_convs[i](out)
        out = self.conv_seg(self.dropout(out))
        return out

class ResNetFeature(nn.Module):
    def __init__(self, n_class):
        super(ResNetFeature, self).__init__()
        self.n_class = n_class
        self.backbone = ResNet('18', 3)

        self.conv0=nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=32,kernel_size=1,stride=1,padding=0),
            # nn.BatchNorm2d(self.dim),
            nn.ReLU(inplace=True),
        )
        self.conv1=nn.Sequential(
            nn.Conv2d(in_channels=128,out_channels=32,kernel_size=1,stride=1,padding=0),
            # nn.BatchNorm2d(self.dim),
            nn.ReLU(inplace=True),
        )
        self.conv2=nn.Sequential(
            nn.Conv2d(in_channels=256,out_channels=32,kernel_size=1,stride=1,padding=0),
            # nn.BatchNorm2d(self.dim),
            nn.ReLU(inplace=True),
        )
        self.conv3=nn.Sequential(
            nn.Conv2d(in_channels=512,out_channels=32,kernel_size=1,stride=1,padding=0),
            # nn.BatchNorm2d(self.dim),
            nn.ReLU(inplace=True),
        )
        # self.head = FPNHead([64, 128, 256, 512], 128, self.n_class)

    def forward(self, x):
        features0,  features1, features2, features3 = self.backbone(x)
        # print(features0.shape,  features1.shape, features2.shape, features3.shape)
        features0,  features1, features2, features3 = self.conv0(features0),  self.conv1(features1), self.conv2(features2), self.conv3(features3)
        # print([feature.shape for feature in features])
        # print(features0.shape,  features1.shape, features2.shape, features3.shape)

        features0 = F.interpolate(features0, size=features0.shape[-2:], mode='bilinear', align_corners=False)
        features1 = F.interpolate(features1, size=features0.shape[-2:], mode='bilinear', align_corners=False)
        features2 = F.interpolate(features2, size=features0.shape[-2:], mode='bilinear', align_corners=False)
        features3 = F.interpolate(features3, size=features0.shape[-2:], mode='bilinear', align_corners=False)

        features = torch.cat([features0,  features1, features2, features3], 1)
        # out = self.head(features)
        return features

if __name__ == '__main__':
    x = torch.randn(32, 3, 256, 256)
    model = ResNetFeature(256)
    print('model parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))
    out = model(x)
    # out = F.interpolate(out, size=x.shape[-2:], mode='bilinear', align_corners=False)
    print(out.shape)
