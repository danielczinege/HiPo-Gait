import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from utils import clones, is_list_or_tuple


class HorizontalPoolingPyramid():
    """
        Horizontal Pyramid Matching for Person Re-identification
        Arxiv: https://arxiv.org/abs/1804.05275
        Github: https://github.com/SHI-Labs/Horizontal-Pyramid-Matching
    """

    def __init__(self, bin_num=None):
        if bin_num is None:
            bin_num = [16, 8, 4, 2, 1]
        self.bin_num = bin_num

    def __call__(self, x):
        """
            x  : [n, c, h, w]
            ret: [n, c, p] 
        """
        n, c = x.size()[:2]
        features = []
        for b in self.bin_num:
            z = x.view(n, c, b, -1)
            z = z.mean(-1) + z.max(-1)[0]
            features.append(z)
        return torch.cat(features, -1)


class SetBlockWrapper(nn.Module):
    def __init__(self, forward_block):
        super(SetBlockWrapper, self).__init__()
        self.forward_block = forward_block

    def forward(self, x, *args, **kwargs):
        """
            In  x: [n, c_in, s, h_in, w_in]
            Out x: [n, c_out, s, h_out, w_out]
        """
        n, c, s, h, w = x.size()
        x = self.forward_block(x.transpose(
            1, 2).reshape(-1, c, h, w), *args, **kwargs)
        output_size = x.size()
        return x.reshape(n, s, *output_size[1:]).transpose(1, 2).contiguous()


class PackSequenceWrapper(nn.Module):
    def __init__(self, pooling_func):
        super(PackSequenceWrapper, self).__init__()
        self.pooling_func = pooling_func

    def forward(self, seqs, seqL, dim=2, options={}):
        """
            In  seqs: [n, c, s, ...]
            Out rets: [n, ...]
        """
        if seqL is None:
            return self.pooling_func(seqs, **options)
        seqL = seqL[0].data.cpu().numpy().tolist()
        start = [0] + np.cumsum(seqL).tolist()[:-1]

        rets = []
        for curr_start, curr_seqL in zip(start, seqL):
            narrowed_seq = seqs.narrow(dim, curr_start, curr_seqL)
            rets.append(self.pooling_func(narrowed_seq, **options))
        if len(rets) > 0 and is_list_or_tuple(rets[0]):
            return [torch.cat([ret[j] for ret in rets])
                    for j in range(len(rets[0]))]
        return torch.cat(rets)


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, bias=False, **kwargs)

    def forward(self, x):
        x = self.conv(x)
        return x


class SeparateFCs(nn.Module):
    def __init__(self, parts_num, in_channels, out_channels, norm=False):
        super(SeparateFCs, self).__init__()
        self.p = parts_num
        self.fc_bin = nn.Parameter(
            nn.init.xavier_uniform_(
                torch.zeros(parts_num, in_channels, out_channels)))
        self.norm = norm

    def forward(self, x):
        """
            x: [n, c_in, p]
            out: [n, c_out, p]
        """
        x = x.permute(2, 0, 1).contiguous()
        if self.norm:
            out = x.matmul(F.normalize(self.fc_bin, dim=1))
        else:
            out = x.matmul(self.fc_bin)
        return out.permute(1, 2, 0).contiguous()


class SeparateBNNecks(nn.Module):
    """
        Bag of Tricks and a Strong Baseline for Deep Person Re-Identification
        CVPR Workshop:  https://openaccess.thecvf.com/content_CVPRW_2019/papers/TRMTMCT/Luo_Bag_of_Tricks_and_a_Strong_Baseline_for_Deep_Person_CVPRW_2019_paper.pdf
        Github: https://github.com/michuanhaohao/reid-strong-baseline
    """

    def __init__(self, parts_num, in_channels, class_num, norm=True, parallel_BN1d=True):
        super(SeparateBNNecks, self).__init__()
        self.p = parts_num
        self.class_num = class_num
        self.norm = norm
        self.fc_bin = nn.Parameter(
            nn.init.xavier_uniform_(
                torch.zeros(parts_num, in_channels, class_num)))
        if parallel_BN1d:
            self.bn1d = nn.BatchNorm1d(in_channels * parts_num)
        else:
            self.bn1d = clones(nn.BatchNorm1d(in_channels), parts_num)
        self.parallel_BN1d = parallel_BN1d

    def forward(self, x):
        """
            x: [n, c, p]
        """
        if self.parallel_BN1d:
            n, c, p = x.size()
            x = x.view(n, -1)  # [n, c*p]
            x = self.bn1d(x)
            x = x.view(n, c, p)
        else:
            x = torch.cat([bn(_x) for _x, bn in zip(
                x.split(1, 2), self.bn1d)], 2)  # [p, n, c]
        feature = x.permute(2, 0, 1).contiguous()
        if self.norm:
            feature = F.normalize(feature, dim=-1)  # [p, n, c]
            logits = feature.matmul(F.normalize(
                self.fc_bin, dim=1))  # [p, n, c]
        else:
            logits = feature.matmul(self.fc_bin)
        return feature.permute(1, 2, 0).contiguous(), logits.permute(1, 2, 0).contiguous()


class BasicConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False, **kwargs):
        super(BasicConv3d, self).__init__()
        self.conv3d = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size,
                                stride=stride, padding=padding, bias=bias, **kwargs)

    def forward(self, ipts):
        '''
            ipts: [n, c, s, h, w]
            outs: [n, c, s, h, w]
        '''
        outs = self.conv3d(ipts)
        return outs
    

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock2D(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock2D, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError(
                'BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError(
                "Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class BasicBlockP3D(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1,  downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlockP3D, self).__init__()
        if norm_layer is None:
            norm_layer2d = nn.BatchNorm2d
            norm_layer3d = nn.BatchNorm3d
        if groups != 1 or base_width != 64:
            raise ValueError(
                'BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError(
                "Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.relu  = nn.ReLU(inplace=True)
        
        self.conv1 = SetBlockWrapper(
            nn.Sequential(
                conv3x3(inplanes, planes, stride), 
                norm_layer2d(planes), 
                nn.ReLU(inplace=True)
            )
        )

        self.conv2 = SetBlockWrapper(
            nn.Sequential(
                conv3x3(planes, planes), 
                norm_layer2d(planes), 
            )
        )

        self.shortcut3d = nn.Conv3d(planes, planes, (3, 1, 1), (1, 1, 1), (1, 0, 0), bias=False)
        self.sbn        = norm_layer3d(planes)

        self.downsample = downsample

    def forward(self, x):
        '''
            x: [n, c, s, h, w]
        '''
        identity = x

        out = self.conv1(x)
        out = self.relu(out + self.sbn(self.shortcut3d(out)))
        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
    
class BasicBlock3D(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=[1, 1, 1],  downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock3D, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        if groups != 1 or base_width != 64:
            raise ValueError(
                'BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError(
                "Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        assert stride[0] in [1, 2, 3]
        if stride[0] in [1, 2]: 
            tp = 1
        else:
            tp = 0
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=(3, 3, 3), stride=stride, padding=[tp, 1, 1], bias=False)
        self.bn1   = norm_layer(planes)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=(3, 3, 3), stride=[1, 1, 1], padding=[1, 1, 1], bias=False)
        self.bn2   = norm_layer(planes)
        self.downsample = downsample

    def forward(self, x):
        '''
            x: [n, c, s, h, w]
        '''
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

def sign_max(array: torch.tensor, dim=0):
    """
        Computes sign max along a given dimension
    """

    abs_array = torch.abs(array)

    # Retrieve the index having the sign max and return the sign max
    max_idx = torch.argmax(abs_array, dim=dim, keepdims=True)
    sign_max_arr = array.gather(dim=dim, index=max_idx)

    # Remove now unused dim
    sign_max_arr = sign_max_arr.squeeze(dim=dim)

    return sign_max_arr

class LayerChannelNorm(nn.Module):
    """
       Reimplemented from torch.nn.LayerNorm.
       It allows to apply layer normalization specifically to the channel dim
    """

    def __init__(self, normalized_shape, channel_dim, num_dims, eps=1e-05, elementwise_affine=True, device=None,
                 dtype=None, **kwargs):
        super(LayerChannelNorm, self).__init__()

        # Private attributes
        self.normalized_shape = normalized_shape
        self.channel_dim = channel_dim
        self.num_dims = num_dims
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.device = device
        self.dtype = dtype
        self.kwargs = kwargs

        # Normal layer norm
        self.layer_norm = nn.LayerNorm(normalized_shape=normalized_shape, eps=eps,
                                       elementwise_affine=elementwise_affine, device=device, dtype=dtype, **kwargs)

    def forward(self, x):

        dim0, dim1 = self.channel_dim, self.num_dims - 1

        # Swap channel dim to last dim and normalize over it
        if dim0 < dim1:
            x1 = x.transpose(dim0, dim1)
            x1 = self.layer_norm(x1)
            x1 = x1.transpose(dim1, dim0)
        else:
            x1 = self.layer_norm(x)

        return x1

    def __repr__(self):
        return f'LayerChannelNorm(normalized_shape={self.normalized_shape}, channel_dim={self.channel_dim}, num_dims={self.num_dims}, eps={self.eps}, elementwise_affine={self.elementwise_affine}, device={self.device}, dtype={self.dtype}{(", " if self.kwargs else "") + ", ".join(str(k) + "=" + str(v) for k, v in self.kwargs.items())})'

class OperationLayer(nn.Module):

    """
        Generic layer that applies an operation on input
        Attributes
        op: function or operation to perform

        Additional parameters are supported
    """

    def __init__(self, op, **kwargs):
        super(OperationLayer, self).__init__()

        self.op = op
        self.kwargs = kwargs

    def forward(self, x, **kwargs):
        return self.op(x, **self.kwargs)

    def __repr__(self):
        return f'{self.__class__.__name__}(op={self.op.__name__}, {", ".join(str(k) + "=" + str(v) for k, v in self.kwargs.items())})'

class WTrainSum(nn.Module):
    """Implementation of the Weighted trainable sum layer proposed in

    Cubero, N., Castro, F.M., Cózar, J.R., Guil, N., Marín-Jiménez, M.J. (2023).
    Multimodal Human Pose Feature Fusion for Gait Recognition.
    In: Pattern Recognition and Image Analysis. IbPRIA 2023.
    Lecture Notes in Computer Science, vol 14062. Springer, Cham.
    https://doi.org/10.1007/978-3-031-36616-1_31

    Arguments
    ---------

    n_branches (int):
        Number of features that will be fused.

    dropout_rate (float):
        Rate of dropout to apply to the attention mask. Default: 0.0.

    softmax (bool):
        Whether to apply softmax to compute the attention mask. Default: true.

    agg_fn (callable):
        Aggregation function to fuse the normalized output. Default: sum (element-wise sum).

    residual_connection (bool):
        Whether to apply a residual connection. Default: true.

    normalize (bool):
        Whether to apply a batch normalization. Default: true.

    channel_dim (int):
        Position of the channel dimension in the input tensor. Default: 1.

    num_dims (int):
        Number of dimensions in the input tensor. Default: 5 (B, C, T, H, W)

    initialization ('ones' or 'uniform'):
        Type of initialization for the attention kernel. Options:

        - Ones: Initialization to 1.
        - Uniform: Random initialization folllowing random uniform sampling.

    """
    init_fns = {'ones': lambda x: None, 'uniform': lambda x: torch.nn.init.uniform_(x, a=-1.0, b=1.0)}
    # Note: No initiallization operation is performed for "ones" as kernel is firstly initialized to 1

    def __init__(self, n_branches, dropout_rate=0.0, softmax=True,
                 agg_fn=torch.sum, residual_connection=True, normalize=True,
                 channel_dim=1, num_dims=5, initialization='ones', **kwargs):
        super(WTrainSum, self).__init__(
            **{key: kwargs[key] for key in kwargs if key not in ('norm_epsilon', 'norm_num_channels')})
        self.n_branches = n_branches
        self.dropout_rate = dropout_rate
        self.softmax = softmax
        self.attention_kernel = None
        self.normalize = normalize
        self.agg_fn = agg_fn
        self.residual_connection = residual_connection
        self.initialization = initialization

        if initialization not in WTrainSum.init_fns:
            raise ValueError(f'initialization must be set to "ones" or "random"')

        # For normalization procedure
        self.norm_layer = None
        self.channel_dim = channel_dim
        self.num_dims = num_dims

        # Build model architecture
        self._build(**kwargs)

    def forward(self, x, **kwargs):
        """Original input data x: [n_branches, B, C, T, H, W]
            Attention mask A: [n_branches]

            output = softmax(A / sqrt(c)) * x
        """
        if self.softmax:
            attention_mask = self.attention_kernel / (self.n_branches ** 0.5)
            attention_mask = F.softmax(attention_mask, dim=0)
        else:
            attention_mask = self.attention_kernel

        if self.dropout_rate > 0.0:
            attention_mask = self.drop_layer(attention_mask)

        # Attention
        ## attention_mat to be reshaped from [n_branches] to [n_branches, 1, 1, 1, 1, 1] so to be mult. by x
        att_exp_dims = attention_mask.size() + ((1,) * (x.dim() - attention_mask.dim()))
        attention_mask = attention_mask.view(att_exp_dims)
        assert attention_mask.size(0) == x.size(0) and all(
            val == 1 for val in attention_mask.size()[1:]), f'Attention mask size {attention_mask.size()}'
        out = attention_mask * x

        # Residual connectin
        if self.residual_connection:
            out = x + out
        # assert self.residual_connection == True

        # Normalize after res. connection
        if self.normalize:
            out = self.norm_layer(out)

        # Aggregation over the dim of n_branches (typically a sum)
        if self.agg_fn:
            out = self.agg_fn(out, **kwargs)

        return out

    def _build(self, **kwargs):

        self.attention_kernel = nn.parameter.Parameter(torch.ones(size=(self.n_branches,), requires_grad=True))

        # Initialize the attention kernel with a different value (omitted for "ones")
        WTrainSum.init_fns[self.initialization](self.attention_kernel)

        if self.dropout_rate > 0.0:
            self.drop_layer = nn.Dropout(self.dropout_rate)

        if self.normalize:
            layer_norm_eps = kwargs.pop('norm_epsilon', 1e-6)
            if 'norm_num_channels' not in kwargs:
                raise ValueError('Number of channels to which normalize should be passed')
            norm_num_channels = kwargs.pop('norm_num_channels')

            # Move channel/filter dim to last axis to normalize over it
            self.norm_layer = LayerChannelNorm(normalized_shape=norm_num_channels, eps=layer_norm_eps,
                                                channel_dim=self.channel_dim + 1, num_dims=self.num_dims + 1)

    def __repr__(self):
        return f'WTrainSum(n_branches={self.n_branches}, dropout_rate={self.dropout_rate}, softmax={self.softmax}, agg_fn={self.agg_fn}, residual_connection={self.residual_connection}, normalize={self.normalize}, channel_dim={self.channel_dim}, num_dims={self.num_dims})\n' \
               f'- weights: {self.attention_kernel}'

class LocalAttentionFusion(nn.Module):
    """Local Attention Fusion module (LAF) from:

    Cubero, N., Castro, F.M., Guil, N., Marín-Jiménez, M.J. (2026).
    HiPo-Gait: Hierarchical Pose model decoupling for Gait Recognition.
    IEEE Transactions on Biometrics, Behavior, and Identity Science.
    https://doi.org/10.1109/TBIOM.2026.3684931

    Arguments
    ---------

    n_branches (int):
        Number of branches that will be fused.

    in_channels (int):
        Number of input channels.

    kernel_size (List[int] ot tuple[int], or int):
        Kernel size for the convolutional layers to compute the attention mask.

    stride (List[int] ot tuple[int], or int):
        Stride for the convolutional layers to compute the attention mask.

    dropout_rate (float):
        Rate of dropout to apply to the attention mask. Default: 0.0.

    softmax (bool):
        Whether to apply softmax to compute the attention mask. Default: true.

    agg_fn (callable):
        Aggregation function to fuse the normalized output. Default: sum (element-wise sum).

    residual_connection (bool):
        Whether to apply a residual connection. Default: true.

    normalize (bool):
        Whether to apply a Layer Norm normalization. Default: true.

    share_weights (bool):
        Whether to share the weights of the convolutional layers across the parallel branches. Default: false

    """

    def __init__(self, n_branches, in_channels, kernel_size=(3, 3, 3),
                 stride=(1, 1, 1), dropout_rate=0.0, softmax=True,
                 agg_fn=torch.sum, residual_connection=True, normalize=True,
                 share_weights=False,
                 **kwargs):
        super(LocalAttentionFusion, self).__init__(
            **{key: kwargs[key] for key in kwargs if key not in ('norm_epsilon',)})
        self.in_channels = in_channels
        self.n_branches = n_branches
        self.kernel_size = kernel_size
        self.stride = stride
        self.dropout_rate = dropout_rate

        if softmax:
            self.softmax = nn.Softmax(dim=0)
            self.use_softmax = True
        else:
            self.softmax = None
            self.use_softmax = False

        self.attention_kernel = None
        self.normalize = normalize
        self.agg_fn = agg_fn
        self.residual_connection = residual_connection
        self.share_weights = share_weights

        # For normalization procedure
        self.norm_layer = None

        # Build layer architecture
        self._build(**kwargs)

    def forward(self, x, **kwargs):
        """Original input data x: [n_branches, B, C, T, H, W]
            Attention mask A: [n_branches, in_channels, kernel_size]

            output = softmax(A(x) / sqrt(c)) * x
        """
        _, B, C, T, H, W = x.size()
        assert x.size(0) == self.n_branches
        # Apply attention mask to all the conv. feature map
        attention_mask = [self.attention_kernel[i](x[i, ...]) for i in range(self.n_branches)]
        attention_mask = torch.stack(attention_mask, dim=0) # [n_branches, B, C, T, H, W]
        assert x.size() == attention_mask.size(), f'Wrong dim for attention mask {attention_mask.size()}. Expected dim: {x.size()}'

        if self.use_softmax:
            attention_mask = attention_mask / (self.n_branches ** 0.5)
            attention_mask = self.softmax(attention_mask)

        if self.dropout_rate > 0.0:
            attention_mask = self.drop_layer(attention_mask)

        out = attention_mask * x

        if self.normalize:
            out = self.norm_layer(out)

        # Apply residual connection
        if self.residual_connection:
            out = x + out

        if self.agg_fn:
            out = self.agg_fn(out, **kwargs)

        return out

    def _build(self, **kwargs):
        if self.share_weights:
            # Create a unique attention kernel for all the patches
            attention_kernel = BasicConv3d(self.in_channels, self.in_channels,
                                                               kernel_size=self.kernel_size, stride=self.stride,
                                                               padding='same')
            self.attention_kernel = nn.ModuleList([attention_kernel for _ in range(self.n_branches)])
        else:
            self.attention_kernel = nn.ModuleList([BasicConv3d(self.in_channels, self.in_channels,
                                                               kernel_size=self.kernel_size, stride=self.stride,
                                                               padding='same') for _ in range(self.n_branches)])

        if self.dropout_rate > 0.0:
            self.drop_layer = nn.Dropout(self.dropout_rate)

        if self.normalize:
            norm_eps = kwargs.pop('norm_epsilon', 1e-6)
            self.norm_layer = LayerChannelNorm(normalized_shape=self.in_channels,
            					eps=norm_eps,
            					channel_dim=2,
                                                num_dims=6)

    def __repr__(self):
        return f'LocalAttentionFusion(n_branches={self.n_branches}, in_channels={self.in_channels}, kernel_size={self.kernel_size}, stride={self.stride}, dropout_rate={self.dropout_rate}, softmax={self.softmax}, agg_fn={self.agg_fn.__name__}, residual_connection={self.residual_connection}, normalize={self.normalize})'

class ConcatConv(nn.Module):
    """Concatenation convolution

    Arguments
    ---------

    n_branches (int):
	Number of branches that will be fused.

    in_channels (int):
	Number of input channels.

    kernel_size (List[int] ot tuple[int], or int):
	Kernel size for the convolutional layers to compute the attention mask.

    stride (List[int] ot tuple[int], or int):
	Stride for the convolutional layers to compute the attention mask.

    dropout_rate (float):
	Rate of dropout to apply to the attention mask. Default: 0.0.

    normalize (bool):
	Whether to apply a Layer Norm normalization. Default: true.

    """

    def __init__(self, n_branches, in_channels, kernel_size=(3, 3, 3),
                 stride=(1, 1, 1), dropout_rate=0.0, normalize=True, **kwargs):
        super(ConcatConv, self).__init__(
            **{key: kwargs[key] for key in kwargs if key not in ('norm_epsilon',)})
        self.in_channels = in_channels
        self.n_branches = n_branches
        self.kernel_size = kernel_size
        self.stride = stride
        self.dropout_rate = dropout_rate


        # Build model architecture
        seq = [BasicConv3d(n_branches * in_channels, in_channels,
                                       kernel_size=kernel_size, stride=stride,
                                       padding='same')]

        if dropout_rate > 0.0:
            seq.append(nn.Dropout(dropout_rate))

        seq.append(nn.LeakyReLU(inplace=True))

        if normalize:
            layer_norm_eps = kwargs.pop('norm_epsilon', 1e-6)

            # Move channel/filter dim to last axis so to normalize over it
            seq.append(LayerChannelNorm(normalized_shape=in_channels, eps=layer_norm_eps,
                                                   channel_dim=1, num_dims=5))


        self.catconv = nn.Sequential(*seq)

    def forward(self, x, **kwargs):
        """Original input data x C [n_branches, B, C, T, H, W]
            Concat and applies fusion conv.
        """
        # Concat. features
        x = x.permute(1, 0, 2, 3, 4, 5)
        x = x.reshape(x.size(0), x.size(1) * x.size(2), x.size(3), x.size(4), x.size(5))

        # Applies fusion conv
        x1 = self.catconv(x)

        return x1

    def __repr__(self):
        return f'ConcatConv(n_branches={self.n_branches}, in_channels={self.in_channels}, kernel_size={self.kernel_size}, stride={self.stride}, dropout_rate={self.dropout_rate})'
