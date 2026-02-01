import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d



class SpatialAttentionModule(nn.Module):
    """
    输入 x ∈ [B,C,H,W]
    1) 在通道维做 avg/max 池化 → [B,1,H,W] & [B,1,H,W]
    2) Cat → [B,2,H,W] → 7×7 Conv → Sigmoid 得到空间注意力图 A ∈ [B,1,H,W]
    3) y = x * A
    4) y → 3×3 Conv + BN + ReLU（不改通道数）
    """
    def __init__(self, in_dim, out_dim, drop_out=False):
        super().__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.post = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
        )
        self.dropout = drop_out

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        attn_in = torch.cat([avgout, maxout], dim=1)  # [B,2,H,W]
        A = self.sigmoid(self.conv2d(attn_in))        # [B,1,H,W]
        y = x * A                                     # 逐点相乘
        y = self.post(y)                              # 3×3 + BN + ReLU
        return y



class DeformableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.offset_conv = nn.Conv2d(in_channels, 2 * kernel_size * kernel_size,
                                     kernel_size=kernel_size, padding=padding)
        self.deform_conv = DeformConv2d(in_channels, out_channels,
                                        kernel_size=kernel_size, padding=padding)
        nn.init.constant_(self.offset_conv.weight, 0)
        nn.init.constant_(self.offset_conv.bias, 0)

    def forward(self, x):
        offset = self.offset_conv(x)
        return self.deform_conv(x, offset)



class Basic_blocks(nn.Module):
    def __init__(self, in_channel, out_channel, decay=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, 1)
        self.conv1 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            DeformableConv2d(out_channel, out_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        return conv2 + x



class DSE_WeightFusion(nn.Module):
    def __init__(self, in_channel, decay=2):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channel, in_channel // decay, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel // decay, in_channel, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        avg_att = self.fc(self.avg_pool(x))
        max_att = self.fc(self.max_pool(x))
        fused_attention = avg_att + max_att
        return x * fused_attention



class ConvBnRelu(nn.Module):
    def __init__(self, in_planes, out_planes, ksize, stride, pad, dilation=1,
                 groups=1, has_bn=True, has_relu=True, inplace=True, has_bias=False):
        super().__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=ksize, stride=stride,
                              padding=pad, dilation=dilation, groups=groups, bias=has_bias)
        self.has_bn = has_bn
        if has_bn:
            self.bn = nn.BatchNorm2d(out_planes)
        self.has_relu = has_relu
        if has_relu:
            self.relu = nn.ReLU(inplace=inplace)

    def forward(self, x):
        x = self.conv(x)
        if self.has_bn:
            x = self.bn(x)
        if self.has_relu:
            x = self.relu(x)
        return x


class AFABlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.dilate1 = nn.Conv2d(in_channels, in_channels, 3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(in_channels, in_channels, 3, dilation=3, padding=3)
        self.dilate3 = nn.Conv2d(in_channels, in_channels, 3, dilation=5, padding=5)
        self.conv1x1 = nn.Conv2d(in_channels, in_channels, 1)
        self.conv = nn.Conv2d(in_channels, in_channels, 1)

        self.se = DSE_WeightFusion(in_channels, decay=2)
        self.conv3x3 = nn.Conv2d(in_channels, 3, 3, padding=1)
        self.conv_last = ConvBnRelu(in_channels, in_channels, 1, 1, 0)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        b1 = self.conv1x1(self.se(self.dilate1(x)))
        b2 = self.conv1x1(self.se(self.dilate2(x)))
        b3 = self.conv1x1(self.se(self.dilate3(x)))
        out = self.relu(self.conv(b1 + b2 + b3))

        att = F.softmax(self.conv3x3(out), dim=1)
        f = att[:, 0:1] * b1 + att[:, 1:2] * b2 + att[:, 2:3] * b3

        ax = self.relu(self.gamma * f + (1 - self.gamma) * x)
        return self.conv_last(ax)



class LayerScale(nn.Module):
    def __init__(self, dim, init_value=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim, 1, 1) * init_value)
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        return x * self.weight + self.bias.view(1, -1, 1, 1)


class _DeformPath(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.offset = nn.Conv2d(c, 18, kernel_size=3, padding=1)
        self.deform = DeformConv2d(c, c, kernel_size=3, padding=1)
        nn.init.constant_(self.offset.weight, 0.)
        nn.init.constant_(self.offset.bias, 0.)

    def forward(self, x):
        return self.deform(x, self.offset(x))


class MDCFBlock(nn.Module):
    def __init__(self, in_channels, ls_init=1e-5, se_decay=2):
        super().__init__()
        assert in_channels % 4 == 0,
        self.subc = in_channels // 4

        # 四个子分支
        self.b1 = nn.Conv2d(self.subc, self.subc, 1, bias=False)
        self.b2 = _DeformPath(self.subc)
        self.b3 = _DeformPath(self.subc)
        self.b4 = _DeformPath(self.subc)

        # 分支拼接后的轻量预处理
        self.bn_pre = nn.BatchNorm2d(in_channels)
        self.act_pre = nn.ReLU(inplace=True)
        self.fuse1x1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        self.bn_fuse = nn.BatchNorm2d(in_channels)
        self.act_fuse = nn.ReLU(inplace=True)

        # 串行双注意力（先通道注意力，再空间注意力）
        self.ca = DSE_WeightFusion(in_channels, decay=se_decay)                  # 通道注意力
        self.sa = SpatialAttentionModule(in_dim=in_channels, out_dim=in_channels)  # 空间注意力(7×7)

        # LayerScale + 输出激活
        self.ls = LayerScale(in_channels, init_value=ls_init)
        self.act_out = nn.ReLU(inplace=True)

    def forward(self, x):
        # 四分支处理
        x1, x2, x3, x4 = torch.chunk(x, 4, dim=1)
        y = torch.cat([self.b1(x1), self.b2(x2), self.b3(x3), self.b4(x4)], dim=1)  # [B,C,H,W]
        y = self.act_pre(self.bn_pre(y))

        # 1×1 融合
        y = self.fuse1x1(y)
        y = self.act_fuse(self.bn_fuse(y))

        # --- 串行双注意力 ---
        y = self.ca(y)  # 先通道注意力
        y = self.sa(y)  # 再空间注意力

        # LayerScale 残差
        y = self.ls(y) + x
        return self.act_out(y)


class MDCDecoderBlock(nn.Module):
    def __init__(self, in_channel, out_channel, hidden_ratio=2, layerscale_init=1e-5):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channel, out_channel, 2, stride=2)
        base_hidden = out_channel * hidden_ratio
        self.hidden_dim = ((base_hidden + 3) // 4) * 4  # 保证被4整除

        self.pre = nn.Sequential(
            nn.Conv2d(out_channel * 2, self.hidden_dim, 1, bias=False),
            nn.BatchNorm2d(self.hidden_dim),
            nn.ReLU(inplace=True),
        )

        self.mdcf = MDCFBlock(self.hidden_dim)

        self.post = nn.Sequential(
            nn.Conv2d(self.hidden_dim, out_channel, 1, bias=False),
            nn.BatchNorm2d(out_channel),
        )
        self.ls = LayerScale(out_channel, init_value=layerscale_init)
        self.act = nn.ReLU(inplace=True)

    def forward(self, high, low):
        up = self.up(high)
        x = torch.cat([up, low], dim=1)
        x = self.post(self.mdcf(self.pre(x)))
        return self.act(self.ls(x) + x)



class MDFANet(nn.Module):
    def __init__(self, n_class=1, input_size=(352, 352)):
        super().__init__()
        self.pool = nn.MaxPool2d(2)

        self.down_conv1 = Basic_blocks(3, 32)
        self.down_conv2 = Basic_blocks(32, 64)
        self.down_conv3 = Basic_blocks(64, 128)
        self.down_conv4 = Basic_blocks(128, 256)
        self.down_conv5 = Basic_blocks(256, 512)

        self.bottleneck_in = nn.Conv2d(512, 1024, 1, bias=False)
        self.down_conv6 = AFABlock(1024)

        self.up_conv5 = MDCDecoderBlock(1024, 512)
        self.up_conv4 = MDCDecoderBlock(512, 256)
        self.up_conv3 = MDCDecoderBlock(256, 128)
        self.up_conv2 = MDCDecoderBlock(128, 64)
        self.up_conv1 = MDCDecoderBlock(64, 32)

        self.dp6 = nn.Conv2d(1024, 1, 1)
        self.dp5 = nn.Conv2d(512, 1, 1)
        self.dp4 = nn.Conv2d(256, 1, 1)
        self.dp3 = nn.Conv2d(128, 1, 1)
        self.dp2 = nn.Conv2d(64, 1, 1)
        self.out = nn.Conv2d(32, 1, 3, padding=1)

        self.center5 = nn.Conv2d(1024, 512, 1)
        self.decodeup4 = nn.Conv2d(512, 256, 1)
        self.decodeup3 = nn.Conv2d(256, 128, 1)
        self.decodeup2 = nn.Conv2d(128, 64, 1)

    def forward(self, inputs):
        b, c, h, w = inputs.size()

        d1 = self.down_conv1(inputs); p1 = self.pool(d1)
        d2 = self.down_conv2(p1);     p2 = self.pool(d2)
        d3 = self.down_conv3(p2);     p3 = self.pool(d3)
        d4 = self.down_conv4(p3);     p4 = self.pool(d4)
        d5 = self.down_conv5(p4);     p5 = self.pool(d5)

        center = self.down_conv6(self.bottleneck_in(p5))

        out6 = F.interpolate(self.dp6(center), (h, w), mode='bilinear', align_corners=False)

        deco5 = self.up_conv5(center, d5)
        out5 = F.interpolate(self.dp5(deco5), (h, w), mode='bilinear', align_corners=False)
        deco5 = deco5 + F.interpolate(self.center5(center), (h // 16, w // 16))

        deco4 = self.up_conv4(deco5, d4)
        out4 = F.interpolate(self.dp4(deco4), (h, w), mode='bilinear', align_corners=False)
        deco4 = deco4 + F.interpolate(self.decodeup4(deco5), (h // 8, w // 8))

        deco3 = self.up_conv3(deco4, d3)
        out3 = F.interpolate(self.dp3(deco3), (h, w), mode='bilinear', align_corners=False)
        deco3 = deco3 + F.interpolate(self.decodeup3(deco4), (h // 4, w // 4))

        deco2 = self.up_conv2(deco3, d2)
        out2 = F.interpolate(self.dp2(deco2), (h, w), mode='bilinear', align_corners=False)
        deco2 = deco2 + F.interpolate(self.decodeup2(deco3), (h // 2, w // 2))

        deco1 = self.up_conv1(deco2, d1)
        out = self.out(deco1)



if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MDCFUNet_NoMDCF(n_class=1, input_size=(256, 256)).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    dummy_input = torch.randn(1, 3, 256, 256).to(device)
    outputs = model(dummy_input)
    print(f"Output shapes: {[out.shape for out in outputs]}")