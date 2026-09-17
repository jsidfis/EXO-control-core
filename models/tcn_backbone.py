import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import torch.nn.functional as F
#深度可分离卷积+自注意力机制+moderntcn大小核分支
class GELU(nn.Module):
    """
    与 torch.nn.GELU 等价的占位类，确保反序列化旧权重时可以找到名称。
    """
    def forward(self, x):
        return F.gelu(x)
class SqueezeExcitation(nn.Module):
    """
    通道注意力模块（Squeeze-and-Excitation）。
    需要在反序列化权重时存在同名类，否则 torch.load 无法找到定义。
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, l = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y
class LayerNorm1d(nn.Module):
    """
    针对 (B, C, L) 输入的 LayerNorm 包装，保证反序列化可用。
    在通道维上做归一化：先将特征维移动到最后，再还原。
    """
    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.channels = channels
        self.ln = nn.LayerNorm(channels, eps=eps)

    def forward(self, x):
        # x: (B, C, L) -> (B, L, C) 以便 LayerNorm 在最后一维归一化
        x_perm = x.transpose(1, 2)
        out = self.ln(x_perm)
        return out.transpose(1, 2)
class MultiScaleTemporalAttention(nn.Module):
    """
    占位/兼容层：用于兼容旧权重中引用的 MultiScaleTemporalAttention。
    - 若模型 state 中包含 branches(ModuleList)，尝试对各分支输出做平均。
    - 否则直接返回输入，保证推理不因缺失定义而中断。
    """
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        branches = getattr(self, "branches", None)
        if branches:
            outs = []
            for m in branches:
                try:
                    outs.append(m(x))
                except Exception:
                    continue
            if outs:
                try:
                    return torch.stack(outs, dim=0).mean(dim=0)
                except Exception:
                    return outs[0]
        return x

class ReparamLargeKernelConv(nn.Module):
    """
    Depthwise large-kernel conv + pointwise (1x1) with optional small-kernel branch.
    Training-time: uses depthwise_large + pointwise + optionally (depthwise_small + pointwise_small).
    Inference-time: call merge_kernel() to fuse branches into a single conv (self.lkb_reparam).
    NOTE: expects input shape (B, C_in, L) and outputs (B, C_out, L).
    """
    def __init__(self, in_channels, out_channels, large_kernel, stride=1,
                 groups=None, small_kernel=None, small_kernel_merged=False, bias=False):
        super().__init__()
        if groups is None:
            groups = in_channels  # default depthwise-like
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.large_kernel = large_kernel
        self.small_kernel = small_kernel
        self.groups = groups
        self.small_kernel_merged = small_kernel_merged

        padding_large = large_kernel // 2
        # depthwise large
        self.depthwise_large = weight_norm(nn.Conv1d(in_channels, in_channels, kernel_size=large_kernel,
                                                     stride=stride, padding=padding_large,
                                                     dilation=1, groups=in_channels, bias=False))
        # pointwise after large
        self.pointwise_large = weight_norm(nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=bias))

        # optional small branch (depthwise small + pointwise small)
        if small_kernel is not None and not small_kernel_merged:
            assert small_kernel <= large_kernel
            padding_small = small_kernel // 2
            self.depthwise_small = weight_norm(nn.Conv1d(in_channels, in_channels, kernel_size=small_kernel,
                                                         stride=stride, padding=padding_small,
                                                         dilation=1, groups=in_channels, bias=False))
            self.pointwise_small = weight_norm(nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=bias))
        elif small_kernel is not None and small_kernel_merged:
            # if pre-merged, we use a single large-kernel conv (initialized later by merge_kernel)
            self.lkb_reparam = nn.Conv1d(in_channels, out_channels, kernel_size=large_kernel,
                                         stride=stride, padding=padding_large, dilation=1,
                                         groups=1, bias=True)  # groups=1, acts as full conv

    def forward(self, x):
        # If merged reparam is present, use it directly
        if hasattr(self, 'lkb_reparam'):
            return self.lkb_reparam(x)

        # large branch
        out_large = self.depthwise_large(x)          # (B, in, L)
        out_large = self.pointwise_large(out_large)  # (B, out, L)

        out = out_large
        # add small branch if present
        if hasattr(self, 'depthwise_small'):
            out_small = self.depthwise_small(x)
            out_small = self.pointwise_small(out_small)
            out = out + out_small
        return out

    def get_equivalent_kernel_bias(self):
        """
        Compute equivalent full-kernel (out_channels, in_channels, K) and bias (out_channels)
        for depthwise+pointwise (+ optional small branch).
        """
        # depthwise_large.weight: (in_channels, 1, K)
        # pointwise_large.weight: (out_channels, in_channels, 1)
        K = self.large_kernel
        device = next(self.parameters()).device

        eq_k = torch.zeros((self.out_channels, self.in_channels, K), device=device)
        eq_b = torch.zeros((self.out_channels,), device=device)

        # large branch contribution
        dw = self.depthwise_large.weight  # (in, 1, K)
        pw = self.pointwise_large.weight  # (out, in, 1)
        # eq_k[o,i,:] = pw[o,i,0] * dw[i,0,:]
        for o in range(self.out_channels):
            for i in range(self.in_channels):
                eq_k[o, i, :] += pw[o, i, 0] * dw[i, 0, :]

        # small branch
        if hasattr(self, 'depthwise_small'):
            dw_s = self.depthwise_small.weight  # (in,1,ks)
            pw_s = self.pointwise_small.weight  # (out,in,1)
            ks = dw_s.shape[-1]
            # pad small to large center
            pad_left = (K - ks) // 2
            pad_right = K - ks - pad_left
            dw_s_padded = F.pad(dw_s, (pad_left, pad_right))  # shape (in,1,K)
            for o in range(self.out_channels):
                for i in range(self.in_channels):
                    eq_k[o, i, :] += pw_s[o, i, 0] * dw_s_padded[i, 0, :]

        return eq_k, eq_b

    def merge_kernel(self):
        """
        Replace branches with a single reparam conv (self.lkb_reparam).
        """
        eq_k, eq_b = self.get_equivalent_kernel_bias()  # eq_b is zeros currently
        # create full conv
        padding = self.large_kernel // 2
        merged = nn.Conv1d(self.in_channels, self.out_channels, kernel_size=self.large_kernel,
                           stride=1, padding=padding, bias=True)
        # merged.weight shape: (out, in, K)
        with torch.no_grad():
            merged.weight.copy_(eq_k)
            merged.bias.copy_(eq_b)
        # attach
        self.lkb_reparam = merged.to(next(self.parameters()).device)
        # delete old modules to avoid duplication
        if hasattr(self, 'depthwise_large'):
            delattr = lambda obj, name: obj.__delattr__(name) if hasattr(obj, name) else None
            delattr(self, 'depthwise_large')
            delattr(self, 'pointwise_large')
        if hasattr(self, 'depthwise_small'):
            delattr(self, 'depthwise_small')
            delattr(self, 'pointwise_small')


class HybridConv1d(nn.Module):
    """
    分支1：原深度可分离卷积 (Depthwise + Pointwise)
    分支2：大核卷积分支 (DepthwiseLarge + Pointwise)
    输出：两分支相加
    """
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1,
                 bias=False, small_kernel=None):
        super().__init__()

        # ---- 分支 1：原始深度可分离卷积 ----
        self.depthwise_sep = DepthwiseSeparableConv1d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation
        )

        # ---- 分支 2：大核深度卷积（需与 dilation 对齐） ----
        effective_kernel = (kernel_size - 1) * dilation + 1

        # depthwise 大核
        self.dw_large = nn.Conv1d(
            in_channels, in_channels,
            kernel_size=effective_kernel,
            stride=stride,
            padding=padding,
            dilation=1,              # 已包含在 effective kernel 里
            groups=in_channels,      # depthwise
            bias=bias
        )

        # pointwise（1x1）
        self.pw_large = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=1,
            bias=bias
        )

        # ---- 可选小核分支 ----
        if small_kernel is not None:
            small_eff = (small_kernel - 1) * dilation + 1
            small_pad = (small_eff - 1)

            self.dw_small = nn.Conv1d(
                in_channels, in_channels,
                kernel_size=small_eff,
                padding=small_pad,
                groups=in_channels,
                bias=bias
            )
            self.pw_small = nn.Conv1d(
                in_channels, out_channels,
                kernel_size=1,
                bias=bias
            )
        else:
            self.dw_small = None

    def forward(self, x):
        # 原分支
        out1 = self.depthwise_sep(x)

        # 大核分支
        out2 = self.pw_large(self.dw_large(x))

        # 小核可选分支
        if self.dw_small is not None:
            out2 = out2 + self.pw_small(self.dw_small(x))

        return out1 + out2


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        """
            # 因果卷积的填充裁剪模块，裁剪掉卷积操作中右侧的多余填充，确保时间卷积的因果性，（即输出只依赖于当前及之前的输入，不依赖未来输入）
        """
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size        # 裁剪尺寸的大小（通常与卷积的 padding 相等）

    def forward(self, x):
        """
        前向传播
        参数:
            x (Tensor): 输入张量，形状为 [batch_size, channels, length]
        返回:
            Tensor: 裁剪后的张量，形状为 [batch_size, channels, length - chomp_size]

            # 定义前向传播逻辑，对输入张量的最后一个维度（即时间维度）进行裁剪右侧部分（去掉多余的填充）
            # [:, :, :-self.chomp_size] 表示：
                第一个维度：所有批次，第二个维度：所有通道，第三个维度：从开始到倒数第chomp_size个元素
                contiguous() 确保内存连续存储，提高后续操作效率
                从输入张量 x 中取时间维度的前 x.size(2) - chomp_size 部分，去掉多余的填充
        """
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):     # TCN 的基本构造模块
    """
        时间卷积块（TCN的基本构建模块），包含两个卷积层，使用残差连接
        因果性裁剪（Chomp1d）：保证卷积结果不依赖未来信息
        激活函数和 Dropout：增加非线性和正则化
        残差连接：缓解梯度消失问题，提升训练稳定性
    """
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        """
        初始化时间卷积块
        参数:
            n_inputs:           输入通道数
            n_outputs:          输出通道数
            kernel_size:        卷积核的大小
            stride:             卷积步长
            dilation:           扩张率，用于扩展感受野（控制感受野大小）
            padding:            填充大小，确保输出序列长度与输入相同;通常为 (kernel_size - 1) * dilation
            dropout:            Dropout 概率，用于正则化
        """
        super(TemporalBlock, self).__init__()

        self.conv1 = HybridConv1d(n_inputs, n_outputs, kernel_size,    # 定义第一个卷积层，并使用 weight_norm 进行权重归一化
                                           stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)      # 创建裁剪模块，去掉右侧多余的填充
        self.relu1 = nn.ReLU()              # ReLU激活函数
        self.dropout1 = nn.Dropout(dropout) # Dropout层，用于正则化

        self.conv2 = HybridConv1d(n_outputs, n_outputs, kernel_size,   # 定义第二个卷积层
                                           stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,    # 将两个卷积层（包括卷积、裁剪、激活、Dropout）组合成序列顺序执行的网络
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None          # 残差连接处理：如果输入和输出通道数不同，使用 1x1 卷积调整通道数
        self.relu = nn.ReLU()   # 残差连接后的激活函数
        self.init_weights()     # 初始化权重

    def init_weights(self):
        """
        兼容初始化：支持原始 DepthwiseSeparableConv1d、我们新增的 HybridConv1d、
        以及可能存在的 ReparamLargeKernelConv 样式（不同属性名）。
        """

        def init_ds_conv(ds):
            # 原 DepthwiseSeparableConv1d (有 depthwise 和 pointwise)
            if hasattr(ds, 'depthwise') and hasattr(ds, 'pointwise'):
                if ds.depthwise is not None:
                    ds.depthwise.weight.data.normal_(0, 0.01)
                if ds.pointwise is not None:
                    ds.pointwise.weight.data.normal_(0, 0.01)

        def init_reparam_like(rp):
            # 兼容不同命名的 reparam / large-kernel 分支
            # 常见字段: depthwise_large, pointwise_large, depthwise_small, pointwise_small
            if hasattr(rp, 'depthwise_large'):
                try:
                    rp.depthwise_large.weight.data.normal_(0, 0.01)
                except Exception:
                    pass
            if hasattr(rp, 'pointwise_large'):
                try:
                    rp.pointwise_large.weight.data.normal_(0, 0.01)
                except Exception:
                    pass
            if hasattr(rp, 'dw_large'):
                try:
                    rp.dw_large.weight.data.normal_(0, 0.01)
                except Exception:
                    pass
            if hasattr(rp, 'pw_large'):
                try:
                    rp.pw_large.weight.data.normal_(0, 0.01)
                except Exception:
                    pass
            if hasattr(rp, 'depthwise_small') and getattr(rp, 'depthwise_small') is not None:
                try:
                    rp.depthwise_small.weight.data.normal_(0, 0.01)
                except Exception:
                    pass
            if hasattr(rp, 'pointwise_small') and getattr(rp, 'pointwise_small') is not None:
                try:
                    rp.pointwise_small.weight.data.normal_(0, 0.01)
                except Exception:
                    pass
            if hasattr(rp, 'dw_small') and getattr(rp, 'dw_small') is not None:
                try:
                    rp.dw_small.weight.data.normal_(0, 0.01)
                except Exception:
                    pass
            if hasattr(rp, 'pw_small') and getattr(rp, 'pw_small') is not None:
                try:
                    rp.pw_small.weight.data.normal_(0, 0.01)
                except Exception:
                    pass

        def init_hybrid(h):
            # HybridConv1d: 初始化内部 depthwise_sep 与 large/small 分支
            if hasattr(h, 'depthwise_sep') and h.depthwise_sep is not None:
                init_ds_conv(h.depthwise_sep)
            # legacy name: reparam (older impl)
            if hasattr(h, 'reparam') and h.reparam is not None:
                init_reparam_like(h.reparam)
            # our newer naming:
            if hasattr(h, 'dw_large') and getattr(h, 'dw_large') is not None:
                try:
                    h.dw_large.weight.data.normal_(0, 0.01)
                except Exception:
                    pass
            if hasattr(h, 'pw_large') and getattr(h, 'pw_large') is not None:
                try:
                    h.pw_large.weight.data.normal_(0, 0.01)
                except Exception:
                    pass
            if hasattr(h, 'dw_small') and getattr(h, 'dw_small') is not None:
                try:
                    h.dw_small.weight.data.normal_(0, 0.01)
                except Exception:
                    pass
            if hasattr(h, 'pw_small') and getattr(h, 'pw_small') is not None:
                try:
                    h.pw_small.weight.data.normal_(0, 0.01)
                except Exception:
                    pass

        # 对 conv1/conv2 做通用初始化（递归检查）
        for conv in (self.conv1, self.conv2):
            # 如果是原 DepthwiseSeparableConv1d
            if isinstance(conv, DepthwiseSeparableConv1d) or (
                    hasattr(conv, 'depthwise') and hasattr(conv, 'pointwise')):
                init_ds_conv(conv)
            # 如果是我们新加的 HybridConv1d
            elif isinstance(conv, HybridConv1d) or hasattr(conv, 'depthwise_sep') or hasattr(conv, 'dw_large'):
                init_hybrid(conv)
            # 如果是直接的 ReparamLargeKernelConv 风格
            elif hasattr(conv, 'depthwise_large') or hasattr(conv, 'pointwise_large') or hasattr(conv,
                                                                                                 'get_equivalent_kernel_bias'):
                init_reparam_like(conv)
            else:
                # 最后兜底：尝试初始化通用 weight 属性（如果存在）
                if hasattr(conv, 'weight'):
                    try:
                        conv.weight.data.normal_(0, 0.01)
                    except Exception:
                        pass

        # downsample（如果存在）
        if self.downsample is not None:
            try:
                self.downsample.weight.data.normal_(0, 0.01)
            except Exception:
                pass

    def forward(self, x):
        """
        前向传播，包含主路径（self.net(x)）和残差连接（x 或 self.downsample(x)）的计算，最终通过 ReLU 激活输出
        参数:
            x (Tensor): 输入张量，形状为 [batch_size, in_channels, seq_len]

        返回:
            Tensor: 输出张量，形状为 [batch_size, out_channels, seq_len]
        """
        out = self.net(x)               # 主路径：通过两个卷积层
        res = x if self.downsample is None else self.downsample(x)  # 残差路径：如果输入输出通道数相同，直接使用输入x；否则通过1x1卷积调整通道数
        return self.relu(out + res)     # 主路径和残差路径相加，然后通过ReLU激活


class DepthwiseSeparableConv1d(nn.Module):
    """深度可分离卷积层"""
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation):
        super().__init__()
        # 深度卷积 (每个输入通道单独卷积)
        self.depthwise = weight_norm(
            nn.Conv1d(in_channels, in_channels, kernel_size,
                      stride=stride, padding=padding, dilation=dilation, groups=in_channels)
        )
        # 逐点卷积 (1x1卷积融合通道信息)
        self.pointwise = weight_norm(
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x):
        return self.pointwise(self.depthwise(x))



class AttentionFusionBlock(nn.Module):
    """
    包含两个并行的、具有不同扩张率的残差块，并融合了通道注意力机制。
    """
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, dropout=0.2):
        """
        初始化注意力时间卷积块
        参数:
            n_inputs:           输入通道数
            n_outputs:          输出通道数
            kernel_size:        卷积核的大小
            stride:             卷积步长
            dilation:           基础扩张率 (d)，用于第一个分支
            dropout:            Dropout 概率
        """
        super(AttentionFusionBlock, self).__init__()

        # 两个扩张率分支
        self.branch1 = TemporalBlock(n_inputs, n_outputs, kernel_size, stride, dilation,
                                     (kernel_size - 1) * dilation, dropout)
        self.branch2 = TemporalBlock(n_inputs, n_outputs, kernel_size, stride, dilation * 2,
                                     (kernel_size - 1) * dilation * 2, dropout)

        # 全局平均池化
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

        # "调整"层：使用FC层生成两个分支的权重
        self.attention_fc = nn.Sequential(
            nn.Linear(n_outputs, n_outputs // 4),  # 使用一个缩减层来降低计算复杂度
            nn.ReLU(inplace=True),
            nn.Linear(n_outputs // 4, n_outputs * 2)  # 输出通道数的两倍，用于两个分支
        )
        self.softmax = nn.Softmax(dim=2) # 在分支维度上应用softmax

        # --- 残差连接 ---
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu_out = nn.ReLU()

    def forward(self, x):
        # 两个并行分支的前向传播
        out1 = self.branch1(x)  # (B, C, L)
        out2 = self.branch2(x)

        # --- 注意力权重生成 ---
        # 1. 逐元素相加初步融合
        merged = out1 + out2    # (B, C, L)

        # 2. 全局池化和FC层调整
        gap = self.global_avg_pool(merged).squeeze(-1)  # (B, C)
        batch_size, num_channels = gap.size()

        # 3. 通过全连接层和softmax得到两个互补的特征描述子，生成两个分支的通道注意力权重
        attention_weights = self.attention_fc(gap).view(batch_size, num_channels, 2) # (B, C, 2)
        attention_weights = self.softmax(attention_weights) # (B, C, 2)

        # --- 特征加权与融合 ---
        # 提取并重塑每个分支的权重（将特征描述子拆分为两个部分）
        w1 = attention_weights[:, :, 0].unsqueeze(-1)  # (B, C, 1)
        w2 = attention_weights[:, :, 1].unsqueeze(-1)  # (B, C, 1)

        # 4. 逐元素相乘得到混合特征
        mixed_feature_1 = out1 * w1
        mixed_feature_2 = out2 * w2

        # 5. 逐元素相加融合混合特征
        fused_out = mixed_feature_1 + mixed_feature_2

        # 最终输出：融合后的特征 + 残差连接，再通过ReLU
        res = x if self.downsample is None else self.downsample(x)
        return self.relu_out(fused_out + res)


class TemporalConvNet(nn.Module):
    """
    完整的时间卷积网络（TCN）由多个AttentionFusionBlock堆叠而成。
    """
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        """
        初始化TCN
        参数:
            num_inputs:     输入通道数 int
            num_channels:   每层的输出通道数列表（隐藏通道数） list
            kernel_size:    卷积核尺寸大小 int
            dropout:        Dropout率 float
        """
        super(TemporalConvNet, self).__init__()
        layers = []     # TCN所有层数，存储所有时间卷积块
        num_levels = len(num_channels)  # 网络层数
        for i in range(num_levels):     # 逐层构建网络
            dilation_size = 2 ** i      # 每层的扩张率（膨胀率），按 2 的幂次递增（1, 2, 4, 8...）
            in_channels = num_inputs if i == 0 else num_channels[i-1]   # 确定当前层的输入通道数：第一层：使用num_inputs，后续层：使用前一层的输出通道数
            out_channels = num_channels[i]      # 当前层的输出通道数
            # 为每一层创建时间卷积块AttentionFusionBlock并添加到layers层列表
            layers += [AttentionFusionBlock(in_channels, out_channels, kernel_size,
                                             stride=1, dilation=dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)   # 将所有层组合成一个顺序执行的网络

    def forward(self, x):
        """
        前向传播
        参数:
            x (Tensor): 输入张量，形状为 [batch_size, in_channels, seq_len]

        返回:
            Tensor: 输出张量，形状为 [batch_size, out_channels[-1], seq_len]
        """
        return self.network(x)  # 直接将输入通过整个网络


class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        """
        初始化TCN模型
        参数:
            input_size (int): 输入特征维度（关节角度数量）
            output_size (int): 输出维度（预测力矩数量）
            num_channels (list): 各层隐藏通道数列表
            kernel_size (int): 卷积核大小
            dropout (float): Dropout概率
        """
        super(TCN, self).__init__()
        # 时间卷积网络（TCN）的核心部分,通过 TemporalConvNet 实现,由多层一维卷积组成的网络，支持因果卷积和扩张卷积,num_channels 决定了每一层卷积的输出通道数
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        # 添加线性层，将TCN网络最后一层的输出映射到目标维度（output_size）
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, x):
        """
        参数:
            输入张量x (Tensor) x.shape = (batch_size, sequence_length, input_size)  batch_size: 批量大小;seq_len: 时间序列长度;input_size: 每个时间步的输入特征维度
            TCN expects input shape: (batch_size, num_features, sequence_length)
        返回:
            输出张量，形状为 [batch_size, output_size]
        """
        # x needs to have dimension (N, C, L) in order to be passed into TCN
        # 维度转换：原始输入维度(N, L, C) : (batch_size, sequence_length, input_size)转为TCN期望输入维度(N, C, L)
        # 通过TCN网络，输出维度: (batch_size, num_channels[-1], sequence_length)
        # 维度还原：将输出转回 (batch_size, sequence_length, num_channels[-1])
        output = self.tcn(x.transpose(1, 2)).transpose(1, 2)        # 将输入从 (N, L, C) 转换为 (N, C, L) 以适应时间卷积的输入要求；output shape: (N, num_channels[-1], L)；将输出从 (N, C, L) 转回 (N, L, C) 以匹配全连接层的输入；output shape: (N, L, num_channels[-1])
        # print(output.size())
        output = self.linear(output)        # 通过线性层，全连接层处理每个时间步的特征；output shape: (N, L, output_size)
        # print(output.size())
        return output[:, -1, :]             # 仅保留序列最后一个时间步的输出（适用于序列到值的任务）；output shape: (N, output_size)


if __name__ == "__main__":
    """
        TCN维度验证
    """
    tcn = TCN(4, 2, [150] * 4, 5, 0.25)

    x = torch.randn(5, 100, 4)          # 生成测试数据形状: (batch_size=5, sequence_length=100, input_size=4)：1个样本，200个时间步，每个时间步4个特征
    print("Input shape:", x.size())
    y = tcn(x)          # 前向传播
    print("Output shape:", y.size())                     # 检查输出形状：torch.Size([5, 2])
    print("Output values:\n", y)            # 打印输出值
