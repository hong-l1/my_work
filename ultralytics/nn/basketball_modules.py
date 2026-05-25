import torch
import torch.nn as nn
from mmcv.ops import DeformConv2d_v3  # 导入MMCV官方优化的DCNv3


class DeformablePerceptionModule(nn.Module):
    """形变感知模块：基于DCNv3自适应捕捉高速运动篮球的形变特征 解决问题：篮球高速运动导致的椭圆拉伸、长条形拖影 输入：YOLOv8 Backbone中P2/4分辨率的C2f特征F_t 输出：增强后的形变特征F_dcn.
    """

    def __init__(self, in_channels, out_channels=None, kernel_size=3, stride=1, padding=1, deform_groups=1):
        super().__init__()
        out_channels = out_channels or in_channels

        self.dcn = DeformConv2d_v3(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=1,
            bias=False,
            deform_groups=deform_groups,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)  # 与YOLOv8保持一致的激活函数

    def forward(self, x):
        x = self.dcn(x)
        x = self.bn(x)
        return self.act(x)


class BGAM(nn.Module):
    """几何先验门控注意力模块：基于篮球尺寸先验过滤背景噪声 解决问题：DCNv3动态感受野引入的球员肢体、观众席等噪声 输入：DCNv3增强特征F_dcn 输出：目标聚焦、高信噪比的特征图.
    """

    def __init__(self, in_channels):
        super().__init__()
        # 尺度感知分支：3x3和5x5并行卷积，严格匹配篮球典型像素尺寸
        self.conv3x3 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1, bias=False)
        self.conv5x5 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=5, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)

        # 空间注意力生成器
        self.attn_conv = nn.Conv2d(in_channels, 1, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

        # 激活函数
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        residual = x  # 残差连接，避免压制弱可见的真实篮球特征

        # 多尺度特征提取
        feat3 = self.conv3x3(x)
        feat5 = self.conv5x5(x)
        feat = torch.cat([feat3, feat5], dim=1)
        feat = self.bn1(feat)
        feat = self.act(feat)

        # 生成空间注意力图M_s
        attn = self.attn_conv(feat)
        attn = self.sigmoid(attn)

        # 残差门控机制：F_out = F_dcn * (1 + M_s)
        # 这种设计既放大了目标区域，又保留了原始特征的完整性
        out = x * attn + residual
        return out
