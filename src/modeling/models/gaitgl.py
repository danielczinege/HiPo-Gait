import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base_model import BaseModel
from ..modules import SeparateFCs, BasicConv3d, PackSequenceWrapper, SeparateBNNecks, LayerChannelNorm


class GLConv(nn.Module):
    """
        Modified version of the original GLConv with added LayerNorm support.
        Check original implementation in: https://github.com/ShiqiYu/OpenGait/blob/master/opengait/modeling/models/gaitgl.py

        Arguments:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            halving (int): Halving factor for local convolution.
            fm_sign (bool): Whether to use feature map sign.
            kernel_size (tuple): Kernel size for convolution.
            stride (tuple): Stride for convolution.
            padding (tuple): Padding for convolution.
            bias (bool): Whether to use bias.
            act_fn (callable): Activation function. Default: Leaky ReLU.
            normalize (bool): Whether to use normalization (Layer Norm). Default: True.
    """
    def __init__(self, in_channels, out_channels, halving, fm_sign=False,
                 kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), 
                 bias=False, act_fn=F.leaky_relu, normalize=True, **kwargs):
        super(GLConv, self).__init__()
        self.halving = halving
        self.fm_sign = fm_sign
        self.global_conv3d = BasicConv3d(
            in_channels, out_channels, kernel_size, stride, padding, bias, **kwargs)
        self.local_conv3d = BasicConv3d(
            in_channels, out_channels, kernel_size, stride, padding, bias, **kwargs)
        self.act_fn = act_fn

        if normalize:
            self.layer_norm = LayerChannelNorm(normalized_shape=out_channels,
                                              channel_dim=1, num_dims=5)
        else:
            self.layer_norm = nn.Identity()

    def forward(self, x):
        '''
            x: [n, c, s, h, w]
        '''
        gob_feat = self.global_conv3d(x)
        if self.halving == 0:
            lcl_feat = self.local_conv3d(x)
        else:
            h = x.size(3)
            split_size = int(h // 2**self.halving)
            lcl_feat = x.split(split_size, 3)
            lcl_feat = torch.cat([self.local_conv3d(_) for _ in lcl_feat], 3)

        if not self.fm_sign:
            gob_feat = self.layer_norm(gob_feat)
            lcl_feat = self.layer_norm(lcl_feat)
            feat = self.act_fn(gob_feat) + self.act_fn(lcl_feat)
        else:
            feat = torch.cat([gob_feat, lcl_feat], dim=3)
            feat = self.layer_norm(feat)
            feat = self.act_fn(feat)
        return feat


class GeMHPP(nn.Module):
    def __init__(self, bin_num=[64], p=6.5, eps=1.0e-6):
        super(GeMHPP, self).__init__()
        self.bin_num = bin_num
        self.p = nn.Parameter(
            torch.ones(1)*p)
        self.eps = eps

    def gem(self, ipts):
        return F.avg_pool2d(ipts.clamp(min=self.eps).pow(self.p), (1, ipts.size(-1))).pow(1. / self.p)

    def forward(self, x):
        """
            x  : [n, c, h, w]
            ret: [n, c, p] 
        """
        n, c = x.size()[:2]
        features = []
        for b in self.bin_num:
            z = x.view(n, c, b, -1)
            z = self.gem(z).squeeze(-1)
            features.append(z)
        return torch.cat(features, -1)


class GaitGL(BaseModel):
    """
        GaitGL: Gait Recognition via Effective Global-Local Feature Representation and Local Temporal Aggregation
        This implementation is modified to apply Layer Normalization to feature maps.
        Reference to original work: https://arxiv.org/pdf/2011.01461.pdf
    """

    def __init__(self, *args, **kargs):
        super(GaitGL, self).__init__(*args, **kargs)

    def build_network(self, model_cfg):
        num_in_channel = model_cfg.get('num_in_channels', 1)
        in_c = model_cfg['channels']
        class_num = model_cfg['class_num']

        # For all datasets
        self.conv3d = nn.Sequential(
            BasicConv3d(num_in_channel, in_c[0], kernel_size=(3, 3, 3),
                        stride=(1, 1, 1), padding=(1, 1, 1)),
            LayerChannelNorm(normalized_shape=in_c[0], channel_dim=1, num_dims=5),
            nn.LeakyReLU(inplace=True)
        )
        self.LTA = nn.Sequential(
            BasicConv3d(in_c[0], in_c[0], kernel_size=(
                3, 1, 1), stride=(3, 1, 1), padding=(0, 0, 0)),
            LayerChannelNorm(normalized_shape=in_c[0], channel_dim=1, num_dims=5),
            nn.LeakyReLU(inplace=True)
        )

        self.GLConvA0 = GLConv(in_c[0], in_c[1], halving=3, fm_sign=False, kernel_size=(
            3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), normalize=True)
        self.MaxPool0 = nn.MaxPool3d(
            kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.GLConvA1 = GLConv(in_c[1], in_c[2], halving=3, fm_sign=False, kernel_size=(
            3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), normalize=True)
        self.GLConvB2 = GLConv(in_c[2], in_c[2], halving=3, fm_sign=True,  kernel_size=(
            3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), normalize=True)

        self.TP = PackSequenceWrapper(torch.max)
        self.HPP = GeMHPP()

        self.Head0 = SeparateFCs(64, in_c[-1], in_c[-1])

        if 'SeparateBNNecks' in model_cfg.keys():
            self.BNNecks = SeparateBNNecks(**model_cfg['SeparateBNNecks'])
            self.Bn_head = False
        else:
            self.Bn = nn.BatchNorm1d(in_c[-1])
            self.Head1 = SeparateFCs(64, in_c[-1], class_num)
            self.Bn_head = True

    def forward(self, inputs):
        ipts, labs, _, _, seqL = inputs
        seqL = None if not self.training else seqL
        if not self.training and len(labs) != 1:
            raise ValueError(
                'The input size of each GPU must be 1 in testing mode, but got {}!'.format(len(labs)))

        # Expand dims if needed
        if len(ipts[0].size()) == 4:
            sils = ipts[0].unsqueeze(1)
        else:
            sils = ipts[0]
            sils = sils.transpose(1, 2).contiguous() # [n, s, j, h, w] -> [n, j, s, h, w]

        del ipts
        n, j, s, h, w = sils.size()
        if s < 3:
            repeat = 3 if s == 1 else 2
            sils = sils.repeat(1, 1, repeat, 1, 1)

        outs = self.conv3d(sils)
        outs = self.LTA(outs)

        outs = self.GLConvA0(outs)
        outs = self.MaxPool0(outs)

        outs = self.GLConvA1(outs)
        outs = self.GLConvB2(outs)  # [n, c, s, h, w]

        outs = self.TP(outs, seqL=seqL, options={"dim": 2})[0]  # [n, c, h, w]
        outs = self.HPP(outs)  # [n, c, p]

        gait = self.Head0(outs)  # [n, c, p]

        if self.Bn_head:  # Original GaitGL Head
            bnft = self.Bn(gait)  # [n, c, p]
            logi = self.Head1(bnft)  # [n, c, p]
            embed = bnft
        else:  # BNNechk as Head
            bnft, logi = self.BNNecks(gait)  # [n, c, p]
            embed = gait

        n, _, s, h, w = sils.size()
        retval = {
            'training_feat': {
                'triplet': {'embeddings': embed, 'labels': labs},
                'softmax': {'logits': logi, 'labels': labs}
            },
            'visual_summary': {
                'image/sils': sils.transpose(1, 2).reshape(n*s, j, h, w)[:, :3]
            },
            'inference_feat': {
                'embeddings': embed
            }
        }
        return retval
