import torch
import torch.nn as nn
import torch.nn.functional as F
# from Res2Net_v1b import res2net50_v1b_26w_4s
from lib.pvtv2 import pvt_v2_b2
from torch.nn import Parameter, init
from torch import einsum
from einops import rearrange
import math
from lib.optim.losses_EDL import *

import torch
import torchvision
from thop import profile

import time
import torch


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_Classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, std=0.001)
        nn.init.constant_(m.bias.data, 0.0)


class Conv1x1(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, 1)
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class Conv_rate(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Conv_rate, self).__init__()
        self.relu = nn.ReLU(True)
        # self.sigmoid = nn.Sigmoid(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4 * out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))
        x = torch.sigmoid(x_cat)

        # x = self.relu(x_cat + self.conv_res(x))
        return x


class CAM_Module(nn.Module):
    """ Channel attention module"""

    # forward
    # 方法中，实现了通道注意力的计算过程。它首先将输入特征图重新排列为(B, C, H * W)
    # 的形状，然后计算了注意力矩阵，即特征图中每个通道与其他通道之间的相关性。接着通过
    # softmax
    # 函数对相关性进行归一化，得到了通道注意力权重。最后，利用计算得到的权重对原始特征图进行加权，得到了加强了通道关注度的特征图。
    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.channel_in = in_dim

        self.gamma = Parameter(torch.zeros(1))
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        max_energy_0 = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)
        energy_new = max_energy_0 - energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        gamma = self.gamma.to(out.device)
        out = gamma * out + x
        return out


class ShallowCAM(nn.Module):

    def __init__(self, feature_dim):
        super(ShallowCAM, self).__init__()
        self.input_feature_dim = feature_dim
        self._cam_module = CAM_Module(self.input_feature_dim)

    def forward(self, x):
        x = self._cam_module(x)

        return x


# 分割任务中实际只用到了z_given_v
class VIB(nn.Module):
    def __init__(self, in_ch=512, z_dim=64, num_class=2):
        super(VIB, self).__init__()
        self.in_ch = in_ch
        self.out_ch = z_dim * 2
        self.num_class = num_class
        self.bottleneck = ChannelCompress(in_ch=self.in_ch, out_ch=self.out_ch)
       

    def forward(self, v):
        z_given_v = self.bottleneck(v)
        # p_y_given_z = self.classifier(z_given_v)
        return z_given_v


class ChannelCompress(nn.Module):
    def __init__(self, in_ch=2048, out_ch=256):
        """
        reduce the amount of channels to prevent final embeddings overwhelming shallow feature maps
        out_ch could be 512, 256, 128
        """
        super(ChannelCompress, self).__init__()
        num_bottleneck = 1000
        add_block = []
        add_block += [nn.Conv2d(in_ch, num_bottleneck, kernel_size=1)]
        add_block += [nn.BatchNorm2d(num_bottleneck)]
        add_block += [nn.ReLU()]

        add_block += [nn.Conv2d(num_bottleneck, 500, kernel_size=1)]
        add_block += [nn.BatchNorm2d(500)]
        add_block += [nn.ReLU()]
        add_block += [nn.Conv2d(500, out_ch, kernel_size=1)]

        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)
        self.model = add_block

    def forward(self, x):
        x = self.model(x)
        return x


class Conv2dbn(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )

        bn = nn.BatchNorm2d(out_channels)

        super(Conv2dbn, self).__init__(conv, bn)


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )
        q = q * self.scale

        sim = einsum("b h d i, b h d j -> b h i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = einsum("b h i j, b h d j -> b h i d", attn, v)
        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
        return self.to_out(out)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, 1), nn.GroupNorm(1, dim))

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
        return self.to_out(out)


class LAtt(nn.Module):
    """Some Information about LAtt"""

    def __init__(self, dim):
        super().__init__()
        self.att = Residual(PreNorm(dim, LinearAttention(dim)))

    def forward(self, x):
        return self.att(x)


class LSAtt(nn.Module):
    '''Linear-Self Attention
    '''

    def __init__(self, dim):
        super().__init__()
        self.s_attn = Residual(PreNorm(dim, Attention(dim)))
        self.l_attn = LAtt(dim)

    def forward(self, x):
        x_s = self.s_attn(x)
        x_l = self.l_attn(x)
        return x_s + x_l


class SDC(nn.Module):
    def __init__(self, in_channels, guidance_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.7):
        super(SDC, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.conv1 = Conv2dbn(guidance_channels, in_channels, kernel_size=3, padding=1)
        self.theta = theta
        self.guidance_channels = guidance_channels
        self.in_channels = in_channels
        self.kernel_size = kernel_size

        self.s_att = LSAtt(in_channels)
        self.conv_rat = Conv_rate(in_channels, in_channels)


    def forward(self, x, guidance):
        guidance = self.conv1(guidance)

        guidance_map = self.s_att(guidance)
        guidance_map_square = torch.square(guidance_map)

        original_map = self.conv_rat(x)

        out = self.conv(guidance_map_square * original_map)

        return out


class SDN(nn.Module):
    def __init__(self, in_channel=3, guidance_channels=2):
        super(SDN, self).__init__()
        self.sdc1 = SDC(in_channel, guidance_channels)

        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(in_channel)

        # self.conv1_3 = Conv1x1(256, 1024)

    def forward(self, feature, guidance):
        boundary_enhanced = self.sdc1(feature, guidance)
        boundary = self.relu(self.bn(boundary_enhanced))
        # feature = self.conv1_3(feature)
        # boundary_enhanced = boundary + feature
        return boundary


class SEAttention(nn.Module):

    def __init__(self, channel=5, reduction=2):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                               bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, input):
        x = self.conv1(input)
        return self.relu(self.bn(x))


class FCA(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.sigmoid = nn.Sigmoid()
        self.in_channels = in_channels
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

    def forward(self, input):
        # global average pooling
        x = self.avgpool(input)
        assert self.in_channels == x.size(1), 'in_channels and out_channels should all be {}'.format(x.size(1))
        x = self.conv(x)
        x = self.sigmoid(x)
        x = torch.mul(input, x)
        return x


class Up_layer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = ConvBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)
        self.fca = FCA(in_channels=out_channels, out_channels=out_channels)

    def forward(self, x, x_aux):
        x = F.interpolate(x, size=x_aux.size()[-2:], mode='bilinear')
        x_aux = self.fca(self.conv(x_aux))
        # x_aux = self.fca(x_aux)
        x = x + x_aux
        return x


class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        bn = nn.BatchNorm2d(out_channels)

        super(Conv2dReLU, self).__init__(conv, bn, relu)


class IBandEDL_plus_pvt(nn.Module):
    # res2net based encoder decoder
    def __init__(self, num_classes=2, channel=32, num_features=2048, in_channel=3, guidance_channels=2):
        super(IBandEDL_plus_pvt, self).__init__()

        self.backbone = pvt_v2_b2()  # [64, 128, 320, 512]
        path = '/home/22031212472/uncertainty/ON/lib/weights/pvt_v2_b2.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)
        # ---- ResNet Backbone ----
        # self.resnet = res2net50_v1b_26w_4s(pretrained=True)
        # ---- Receptive Field Block like module ----
        # self.rfb2_1 = RFB_modified(512, channel)
        # self.rfb3_1 = RFB_modified(1024, channel)
        # self.rfb4_1 = RFB_modified(2048, channel)

        self.num_classes = num_classes
        self.in_planes = num_features

        self.cam_256 = ShallowCAM(256)
        self.cam_512 = ShallowCAM(512)
        self.cam_1024 = ShallowCAM(1024)

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.in_planes = 2048
        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck2 = nn.BatchNorm2d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)

        # classifier = [nn.Linear(num_features, num_classes)]
        # classifier = nn.Sequential(*classifier)
        # self.classifier = classifier
        # self.bottleneck.apply(weights_init_kaiming)
        # self.classifier.apply(weights_init_classifier)

        self.z_dim = 64

        self.RGB_Bottleneck = VIB(in_ch=512, z_dim=self.z_dim, num_class=num_classes)

        self.conv1_1 = Conv1x1(1024, 256)
        self.conv1_2 = Conv1x1(2048, 512)
        self.conv1_3 = Conv1x1(256, 1024)
        self.conv1_4 = Conv1x1(512, 256)
        self.conv1_5 = Conv1x1(5, 3)
        self.conv1_6 = Conv1x1(512, 1)

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.up8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)

        self.down = nn.MaxPool2d(kernel_size=2, stride=2)

        self.sdn1 = SDN(320, 512)
        self.sdn2 = SDN(128, 320)
        self.sdn3 = SDN(64, 128)

        # self.sdc = SDC(in_channel, guidance_channels)

        self.se = SEAttention(channel=5, reduction=2)

        # self.scale_1 = nn.Conv2d(in_channels=512, out_channels=)

        self.softplus = nn.Softplus()

        self.conv_r = ConvBlock(in_channels=128, out_channels=64, kernel_size=1, stride=1, padding=0)

        self.up_d1 = Up_layer(in_channels=320, out_channels=64)
        self.up_d2 = Up_layer(in_channels=128, out_channels=64)
        self.up_d3 = Up_layer(in_channels=64, out_channels=64)

        # self.convd1 = ConvBlock(512, 256)
        # self.convd2 = ConvBlock(256, 64)

        self.scale_0 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1)
        self.scale_1 = nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=1)

        channels = [64, 128, 320, 512]

        self.decoder1 = nn.Sequential(
            Conv2dReLU(channels[3] + channels[2], channels[2], kernel_size=3, padding=1),
            Conv2dReLU(channels[2], channels[2], kernel_size=3, padding=1)
        )

        self.decoder2 = nn.Sequential(
            Conv2dReLU(channels[2] + channels[1], channels[1], kernel_size=3, padding=1),
            Conv2dReLU(channels[1], channels[1], kernel_size=3, padding=1)
        )

        self.decoder3 = nn.Sequential(
            Conv2dReLU(channels[1] + channels[0], channels[0], kernel_size=3, padding=1),
            Conv2dReLU(channels[0], 1, kernel_size=3, padding=1)
        )

        # self.num_classes = num_classes

        # self.init_weight()

        # self.loss_fn = bce_iou_loss

    def DS_Combin(self, alpha):
        """
        :param alpha: All Dirichlet distribution parameters.
        :return: Combined Dirichlet distribution parameters.
        """

        def DS_Combin_two(alpha1, alpha2):
            """
            :param alpha1: Dirichlet distribution parameters of view 1
            :param alpha2: Dirichlet distribution parameters of view 2
            :return: Combined Dirichlet distribution parameters
            """
            alpha = dict()
            alpha[0], alpha[1] = alpha1, alpha2
            b, S, E, u = dict(), dict(), dict(), dict()
            for v in range(2):
                S[v] = torch.sum(alpha[v], dim=1, keepdim=True)
                E[v] = alpha[v] - 1
                b[v] = E[v] / (S[v].expand(E[v].shape))
                u[v] = self.num_classes / S[v]

            # b^0 @ b^(0+1)
            bb = torch.bmm(b[0].view(-1, self.num_classes, 1), b[1].view(-1, 1, self.num_classes))
            # b^0 * u^1
            uv1_expand = u[1].expand(b[0].shape)
            bu = torch.mul(b[0], uv1_expand)
            # b^1 * u^0
            uv_expand = u[0].expand(b[0].shape)
            ub = torch.mul(b[1], uv_expand)
            # calculate C
            bb_sum = torch.sum(bb, dim=(1, 2), out=None)
            bb_diag = torch.diagonal(bb, dim1=-2, dim2=-1).sum(-1)
            # bb_diag1 = torch.diag(torch.mm(b[v], torch.transpose(b[v+1], 0, 1)))
            C = bb_sum - bb_diag

            # calculate b^a
            b_a = (torch.mul(b[0], b[1]) + bu + ub) / ((1 - C).view(-1, 1).expand(b[0].shape))
            # calculate u^a
            u_a = torch.mul(u[0], u[1]) / ((1 - C).view(-1, 1).expand(u[0].shape))

            # calculate new S
            S_a = self.num_classes / u_a
            # calculate new e_k
            e_a = torch.mul(b_a, S_a.expand(b_a.shape))
            alpha_a = e_a + 1
            return alpha_a

        for v in range(len(alpha) - 1):
            if v == 0:
                alpha_a = DS_Combin_two(alpha[0], alpha[1])
            else:
                alpha_a = DS_Combin_two(alpha_a, alpha[v + 1])
        return alpha_a

    def forward(self, sample, bs, epoch):
    # def forward(self, x):

        x = sample['image']
        if 'gt' in sample.keys():
            y = sample['gt']
        else:
            y = None

        x0 = x
        pvt = self.backbone(x)
        x1 = pvt[0]   # bs, 64, 88, 88
        x2 = pvt[1]     # bs, 128, 44, 44
        x3 = pvt[2]     # bs, 320, 22, 22
        x4 = pvt[3]     # bs, 512, 11, 11

        
        x2_1 = self.cam_256(x2)
        x3_1 = pvt[2]

        
        x3_1 = self.cam_512(x3_1)

        x4_1 = pvt[3]


        
        x4_1 = self.cam_1024(x4_1)

        ib_1 = self.RGB_Bottleneck(x4_1)  # bs, 128, 11 ,11 通过decoder

        ib_1 = self.conv_r(ib_1)        # bs, 32, 11 ,11

        x_d2 = self.up_d1(ib_1, x3)     # bs, 64, 22 ,22
        # x_d2 = self.convd1(x_d2)
        x_d1 = self.up_d2(x_d2, x2)     # bs, 64, 44, 44
        # x_d1 = self.convd2(x_d1)
        x_d0 = self.up_d3(x_d1, x1)     # bs, 64, 88, 88

        # -----------------first-branch-result------------------------

        x_final_0 = self.scale_0(x_d0)
        x_final_0 = F.interpolate(x_final_0, size=x0.size()[-2:], mode='bilinear')

        x_final_1 = self.scale_1(x_d0)
        x_final_1 = F.interpolate(x_final_1, size=x0.size()[-2:], mode='bilinear')

        be = F.interpolate(x4, size=x0.size()[-2:], mode='bilinear')

        boundary_enhanced = self.conv1_6(be)

        # x4是deep_feature
        # x3是current_feataure

        # -----------------boundary-enhanced-features------------------

        df1 = F.interpolate(x4, size=x3.size()[-2:], mode='bilinear')
        cf1 = x3
        o1 = self.sdn1(cf1, df1)
        be1 = o1 + cf1
        be1 = torch.cat((df1, be1), 1)
        d1 = self.decoder1(be1)  # (bs, 320, 22, 22)

        df2 = F.interpolate(d1, size=x2.size()[-2:], mode='bilinear')
        cf2 = x2
        o2 = self.sdn2(cf2, df2)
        be2 = o2 + cf2
        be2 = torch.cat((df2, be2), 1)
        d2 = self.decoder2(be2)

        df3 = F.interpolate(d2, size=x1.size()[-2:], mode='bilinear')
        cf3 = x1
        o3 = self.sdn3(cf3, df3)
        be3 = o3 + cf3
        be3 = torch.cat((df3, be3), 1)
        d3 = self.decoder3(be3)  # (bs,256,88,88)

        boundary_enhanced = F.interpolate(d3, size=x0.size()[-2:], mode='bilinear')

        second_i = torch.cat((x0, boundary_enhanced, x_final_0), dim=1)
        second_i = self.se(second_i)

        second_i = self.conv1_5(second_i)

        pvt = self.backbone(second_i)
        x5 = pvt[0]
        x6 = pvt[1]
        x7 = pvt[2]
        x8 = pvt[3]
        
        x6_1 = self.cam_256(x6)

        # x8 = self.resnet.layer3(x7_1)
        x7_1 = self.cam_512(x7)

        # x9 = self.resnet.layer4(x8_1)
        x8_1 = self.cam_1024(x8)

        ib_2 = self.RGB_Bottleneck(x8_1)  # bs, 128, 11, 11 通过decoder

        ib_2 = self.conv_r(ib_2)

        x_d3 = self.up_d1(ib_2, x3)
        # x_d3 = self.convd1(x_d3)
        x_d4 = self.up_d2(x_d3, x2)
        # x_d4 = self.convd2(x_d4)
        x_d5 = self.up_d3(x_d4, x1)

        # -----------------second-branch-result------------------------

        x_final_2 = self.scale_1(x_d5)
        x_final_2 = F.interpolate(x_final_2, size=x0.size()[-2:], mode='bilinear')

        e_1 = self.softplus(x_final_1)
        e_2 = self.softplus(x_final_2)

        evidence = dict()
        evidence[0] = e_1.permute(0, 2, 3, 1).reshape(-1, self.num_classes)  # (B*H*W, 512)
        evidence[1] = e_2.permute(0, 2, 3, 1).reshape(-1, self.num_classes)  # (B*H*W, 512)

        alpha = dict()
        for v_num in range(len(evidence)):
            alpha[v_num] = evidence[v_num] + 1

        alpha_a = self.DS_Combin(alpha)
        evidence_a = alpha_a - 1

        

        # return evidence_a

        if y is not None:
            # y = y.flatten()
            # loss1 = self.loss_fn(pred, y)
            loss2 = 0
            y = y.flatten()
            for v_num in range(len(evidence)):
                loss2 += ce_loss(y, alpha[v_num], self.num_classes, epoch, 50)

            loss2 += 2 * ce_loss(y, alpha_a, self.num_classes, epoch, 50)

            loss = loss2

            loss = torch.mean(loss)



        else:
            loss = 0

        return {'loss': loss,
                'alpha_a': alpha_a,
                'evidence': evidence,
                'evidence_a': evidence_a,
                'alpha': alpha
                # 'be': second_i

                }
    
# if __name__ == '__main__':
#     # -- coding: utf-8 --
    

#     # Model
#     print('==> Building model..')
#     model = IBandEDL_plus_pvt().cuda()

#     dummy_input = torch.randn(1, 3, 352, 352).cuda()
#     flops, params = profile(model, (dummy_input,))
#     print('flops: ', flops, 'params: ', params)
#     print('flops: %.2f G, params: %.2f M' % (flops / 1000000000.0, params / 1000000.0))

    

#     input_tensor = torch.randn(1, 3, 352, 352).cuda()


#     # 推断的次数
#     num_inferences = 100  # 假设进行100次推断

#     # 加载模型到 GPU
#     model = IBandEDL_plus_pvt().cuda()
    
#     # 确保 CUDA 操作的同步
#     torch.cuda.synchronize()
    
#     # 开始计时
#     start = time.time()
    
#     # 进行多次推断
#     for _ in range(num_inferences):
#         result = model(input_tensor).cuda()  # 重新加载模型以模拟多次推断
#         # print(result)

#     # result = model(input_tensor).cuda()
    
#     # 确保 CUDA 操作的同步
#     torch.cuda.synchronize()
    
#     # 结束计时
#     end = time.time()
    
#     # 计算总时间
#     total_time = end - start
    
#     # 计算每秒钟的推断次数
#     fps = num_inferences / total_time
    
#     print('FPS:', fps)





