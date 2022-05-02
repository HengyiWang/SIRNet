from torch import nn
import torch.utils.model_zoo as model_zoo
import torch


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def Upsample(x, size):
    """
    Wrapper Around the Upsample Call
    """
    return nn.functional.interpolate(x, size=size, mode='bilinear',
                                     align_corners=False)


def scale_as(x, y):
    '''
    scale x to the same size as y
    '''
    y_size = y.size(2), y.size(3)
    x_scaled = torch.nn.functional.interpolate(
        x, size=y_size, mode='bilinear', align_corners=False)
    return x_scaled


class Bottleneck(nn.Module):
    """
    Bottleneck Layer for Resnet
    """
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """
    Resnet Global Module for Initialization
    """

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
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
        for index in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet50(pretrained=True, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


class Backbone(nn.Module):
    def __init__(self, trunk_name='resnet-50', output_stride=8):
        super(Backbone, self).__init__()

        if trunk_name == 'resnet-50':
            resnet = resnet50()
            resnet.layer0 = nn.Sequential(resnet.conv1, resnet.bn1,
                                          resnet.relu, resnet.maxpool)
        else:
            raise ValueError("Not a valid network arch")

        self.layer0 = resnet.layer0
        self.layer1, self.layer2, self.layer3, self.layer4 = \
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

        if output_stride == 8:
            for n, m in self.layer3.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)
            for n, m in self.layer4.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)
        elif output_stride == 16:
            for n, m in self.layer4.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)
        else:
            raise 'unsupported output_stride {}'.format(output_stride)

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        s2_features = x
        x = self.layer2(x)
        s4_features = x
        x = self.layer3(x)
        x = self.layer4(x)
        return s2_features, s4_features, x


class AtrousSpatialPyramidPoolingModule(nn.Module):
    """
    operations performed:
      1x1 x depth
      3x3 x depth dilation 6
      3x3 x depth dilation 12
      3x3 x depth dilation 18
      image pooling
      concatenate all together
      Final 1x1 conv
    """

    def __init__(self, in_dim, reduction_dim=256, output_stride=16,
                 rates=(6, 12, 18)):
        super(AtrousSpatialPyramidPoolingModule, self).__init__()

        if output_stride == 8:
            rates = [2 * r for r in rates]
        elif output_stride == 16:
            pass
        else:
            raise 'output stride of {} not supported'.format(output_stride)

        self.features = []
        # 1x1
        self.features.append(
            nn.Sequential(nn.Conv2d(in_dim, reduction_dim, kernel_size=1,
                                    bias=False),
                          nn.BatchNorm2d(reduction_dim), nn.ReLU(inplace=True)))
        # other rates
        for r in rates:
            self.features.append(nn.Sequential(
                nn.Conv2d(in_dim, reduction_dim, kernel_size=3,
                          dilation=r, padding=r, bias=False),
                nn.BatchNorm2d(reduction_dim),
                nn.ReLU(inplace=True)
            ))
        self.features = nn.ModuleList(self.features)

        # img level features
        self.img_pooling = nn.AdaptiveAvgPool2d(1)
        self.img_conv = nn.Sequential(
            nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False), nn.ReLU(inplace=True))
            #nn.BatchNorm2d(reduction_dim),

    def forward(self, x):
        x_size = x.size()
        img_features = self.img_pooling(x)
        img_features = self.img_conv(img_features)
        img_features = Upsample(img_features, x_size[2:])
        out = img_features

        for f in self.features:
            y = f(x)
            out = torch.cat((out, y), 1)
        return out

class MsOSVOS(nn.Module):
    def __init__(self, scales=(0.5, 1.0), s2_ch=256, high_level_ch=2048, bottle_ch=256, output_stride=8, num_classes=2):
        super(MsOSVOS, self).__init__()
        self.backbone = Backbone()
        self.aspp = AtrousSpatialPyramidPoolingModule(high_level_ch, bottle_ch,
                                                 output_stride=output_stride)
        self.bot_fine = nn.Conv2d(s2_ch, 48, kernel_size=1, bias=False)
        self.bot_aspp = nn.Conv2d(5 * bottle_ch, 256, kernel_size=1, bias=False)
        self.final = nn.Sequential(
            nn.Conv2d(256 + 48, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1, bias=False))

        # Scale-attention prediction head
        scale_in_ch = 2 * (256 + 48)

        self.scale_attn = nn.Sequential(
            nn.Conv2d(scale_in_ch, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1, kernel_size=1, bias=False),
            nn.Sigmoid())

        self.kl_distance = nn.KLDivLoss(reduction='none')
        self.sm = torch.nn.Softmax(dim=1)
        self.log_sm = torch.nn.LogSoftmax(dim=1)

        self.scales = scales

    def _fwd(self, x):
        x_size = x.size()
        s2_features, _, final_features = self.backbone(x)  # s2 s4 s8
        aspp = self.aspp(final_features)  # backbone and the aspp are all about the encoder  aspp = 1/64

        conv_aspp = self.bot_aspp(aspp)  # 1/64 256 channels 路线1
        conv_s2 = self.bot_fine(s2_features)  # 1/2 48 channels 路线2
        conv_aspp = Upsample(conv_aspp, s2_features.size()[2:])  # 1/2 256 channels 路线1
        cat_s4 = [conv_s2, conv_aspp]
        cat_s4 = torch.cat(cat_s4, 1)  # 1/2 256+48 channels r1

        final = self.final(cat_s4)  # 1/2 结果 channels r1
        out = Upsample(final, x_size[2:])  # 1 1 channels r1

        out = torch.softmax(out, dim=1)

        return out, cat_s4

    def two_scale_forward(self, inputs):
        x_1x = inputs['1.0']
        x_lo = inputs['0.5']
        p_1x, feats_hi = self._fwd(x_1x)
        p_lo, feats_lo = self._fwd(x_lo)

        variance = torch.sum(self.kl_distance(
            self.log_sm(scale_as(p_lo, p_1x)), self.sm(p_1x)), dim=1)

        feats_hi = scale_as(feats_hi, feats_lo)
        cat_feats = torch.cat([feats_lo, feats_hi], 1)
        logit_attn = self.scale_attn(cat_feats)
        logit_attn = scale_as(logit_attn, p_lo)

        p_lo = logit_attn * p_lo
        p_lo = scale_as(p_lo, p_1x)
        logit_attn = scale_as(logit_attn, p_1x)
        joint_pred = p_lo + (1 - logit_attn) * p_1x

        outputs = torch.softmax(joint_pred, dim=1)

        # attn [bs, c=1, h, w]
        # Var [bs, h, w]
        return outputs, logit_attn, variance

    def nscale_forward(self, inputs, scales):
        x_1x = inputs['1.0']

        assert 1.0 in scales, 'expected 1.0 to be the target scale'
        # Lower resolution provides attention for higher rez predictions,
        # so we evaluate in order: high to low
        scales = sorted(scales, reverse=True)
        pred = None
        last_feats = None

        for idx, s in enumerate(scales):
            x = inputs[str(s)]
            p, feats = self._fwd(x)

            # Generate attention prediction
            if idx > 0:
                assert last_feats is not None
                # downscale feats
                last_feats = scale_as(last_feats, feats)
                cat_feats = torch.cat([feats, last_feats], 1)
                attn = self.scale_attn(cat_feats)
                attn = scale_as(attn, p)

            if pred is None:
                # This is the top scale prediction
                pred = p
            elif s >= 1.0:
                # downscale previous
                pred = scale_as(pred, p)
                pred = attn * p + (1 - attn) * pred
            else:
                # upscale current
                p = attn * p
                p = scale_as(p, pred)
                attn = scale_as(attn, pred)
                pred = p + (1 - attn) * pred

            last_feats = feats

        outputs = torch.softmax(pred, dim=1)

        return outputs, attn

    def forward(self, inputs, fwd=False, nscale=False, scales=(0.5, 1.0, 1.5)):
        if fwd:
            return self._fwd(inputs)
        if nscale:
            return self.nscale_forward(inputs, scales=scales)
        return self.two_scale_forward(inputs)


