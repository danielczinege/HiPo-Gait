import sys
from itertools import chain
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base_model import BaseModel
from ..modules import (SeparateFCs, BasicConv3d, PackSequenceWrapper, SeparateBNNecks,
                        LayerChannelNorm, LocalAttentionFusion, WTrainSum, sign_max,
                        OperationLayer, ConcatConv)
from utils import torch_max

class MetaHiPoGait(BaseModel):
    """
        Meta definition of the HiPo-Gait model
    """

    def __init__(self, *args, **kargs):
        super(MetaHiPoGait, self).__init__(*args, **kargs)

    # Core builder function
    def build_network(self, model_cfg):
        #in_c = model_cfg['channels']
        self.out_channel_block = [None, None, None]
        dataset_name = self.cfgs['data_cfg']['dataset_name']


        self.body_part_channels = model_cfg.get('body_part_channels',
                                    dict(zip(('left_arm', 'left_leg',
                                              'right_arm', 'right_leg'),
                                                            list(range(4)))))

        self.agg_method = model_cfg['aggregation_method']

        # Parameters for WTrainSum and LAF
        aggregation_method_cfg = model_cfg.get('aggregation_method_cfg', {})

        """
            For body_parts_channel. Expected input is:
            - left_leg: 0
            - right_leg: 1
            - left_arm: 2
            - right_arm: 3
        """
        self.body_part_channels = {key: int(self.body_part_channels[key]) for key in self.body_part_channels}

        ### Make 4 branches for hierarchical blocks 1, one per each body part
        self.HBs1 = nn.ModuleList([self.build_hierblock1(model_cfg=model_cfg, dataset_name=dataset_name) for _ in range(4)])

        ##  First fusion stage: left leg with right leg and left arm with right arm.
        self.fuse_lay0 = self.make_fusion_stage(num_branches=2, in_c=self.out_channel_block[0], cfg=aggregation_method_cfg)

        ## Make 2 branches for hierarchical blocks 2
        self.HBs2 = nn.ModuleList([self.build_hierblock2(model_cfg=model_cfg,
                                                   dataset_name=dataset_name
                                                   ) for _ in range(2)])

        ##  Second fusion stage: left leg + right leg with left arm + right arm
        self.fuse_lay1 = self.make_fusion_stage(num_branches=1, in_c=self.out_channel_block[1], cfg=aggregation_method_cfg)

        ## Make 1 branch for hierarchical block3
        self.HB3 = self.build_hierblock3(model_cfg=model_cfg,
                                                   dataset_name=dataset_name)

    # Model network builders
    def build_hierblock1(self, **kwargs):
        raise NotImplementedError()

    def build_hierblock2(self, **kwargs):
        raise NotImplementedError()

    def build_hierblock3(self, **kwargs):
        raise NotImplementedError()

    def make_fusion_stage(self, num_branches, in_c, cfg, num_in_branches=2):
        fuse_lay = None

        if self.agg_method == 'WTrainSum':
            fuse_lay = [WTrainSum(n_branches=num_in_branches,
                                    dropout_rate=cfg.get('dropout_rate', 0.0),
                                    agg_fn=torch.sum,
                                    softmax=cfg.get('softmax', True),
                                    residual_connection=cfg.get('residual_connection', True),
                                    normalize=cfg.get('layer_norm', True),
                                    norm_num_channels=in_c,
                                    norm_epsilon=cfg.get('layer_norm_eps', 1e-5),
                                    channel_dim=1,
                                    num_dims=5) for _ in range(num_branches)]
        elif self.agg_method == 'LAF':
            fuse_lay = [LocalAttentionFusion(n_branches=num_in_branches,
                                          in_channels=in_c,
                                          kernel_size=cfg.get('kernel_size', (3, 3, 3)),
                                          stride=cfg.get('stride', (1, 1, 1)),
                                          dropout_rate=cfg.get('dropout_rate', 0.0),
                                          agg_fn=torch.sum,
                                          softmax=cfg.get('softmax', True), 
                                          residual_connection=cfg.get('residual_connection', True),
                                          normalize=cfg.get('layer_norm', True), 
                                          norm_epsilon=cfg.get('layer_norm_eps', 1e-5)
                                        ) for _ in range(num_branches)]
        elif self.agg_method == 'concat':
            fuse_lay = [ConcatConv(n_branches=num_in_branches,
                                          in_channels=in_c,
                                          kernel_size=cfg.get('kernel_size', (3, 3, 3)),
                                          stride=cfg.get('stride', (1, 1, 1)),
                                          normalize=cfg.get('layer_norm', True),
                                          norm_epsilon=cfg.get('layer_norm_eps', 1e-5)
                                        ) for _ in range(num_branches)]
        elif self.agg_method == 'sum':
            fuse_lay = [OperationLayer(torch.sum, dim=0) for _ in range(num_branches)]
        elif self.agg_method == 'max':
            fuse_lay = [OperationLayer(torch_max, dim=0) for _ in range(num_branches)]
        elif self.agg_method == 'mean':
            fuse_lay = [OperationLayer(torch.mean, dim=0) for _ in range(num_branches)]
        elif self.agg_method == 'sign_max':
            fuse_lay = [OperationLayer(sign_max, dim=0) for _ in range(num_branches)]
        else:
            raise ValueError('Invalid aggregation method: {}'.format(self.agg_method))

        # Rejoint stage layers
        if num_branches == 1:
            fuse_lay = fuse_lay[0]
        else:
            fuse_lay = nn.ModuleList(fuse_lay)

        return fuse_lay


    def forward(self, inputs):
        ipts, labs, typs, vies, seqL = inputs

        # Collect all the extra args
        kwargs = {
                    'labs': labs,
                    'typs': typs,
                    'vies': vies,
                    'seqL': seqL
            }

        # Ensure input is in format [n, j, s, h, w]
        if len(ipts[0].size()) == 4:
            sils = ipts[0].unsqueeze(1)
        else:
            sils = ipts[0]
            sils = sils.transpose(1, 2).contiguous()

        del ipts

        n, j, s, h, w = sils.size()
        n_prev, j_prev, s_prev, h_prev, w_prev = n, j, s, h, w

        # Isolate each body part
        br_idx_l_leg = self.body_part_channels['left_leg']
        br_idx_r_leg = self.body_part_channels['right_leg'] 
        br_idx_l_arm = self.body_part_channels['left_arm'] 
        br_idx_r_arm = self.body_part_channels['right_arm']

        sils_left_leg = sils[:, br_idx_l_leg: br_idx_l_leg + 1]
        sils_right_leg = sils[:, br_idx_r_leg: br_idx_r_leg + 1]
        sils_left_arm = sils[:, br_idx_l_arm: br_idx_l_arm+ 1]
        sils_right_arm = sils[:, br_idx_r_arm: br_idx_r_arm + 1]


        ## Input Block 1 pass
        outs = [
                self.feed_hierblock1(self.HBs1[0], sils_left_leg, **kwargs),
                self.feed_hierblock1(self.HBs1[1], sils_right_leg, **kwargs),
                self.feed_hierblock1(self.HBs1[2], sils_left_arm, **kwargs),
                self.feed_hierblock1(self.HBs1[3], sils_right_arm, **kwargs)
               ]

        ### First fusion: left leg with right leg and left arm with right arm
        outs = [
                self.feed_fusion(self.fuse_lay0[0], [outs[0], outs[1]], **kwargs), # left leg + right leg
                self.feed_fusion(self.fuse_lay0[1], [outs[2], outs[3]], **kwargs), # left arm + right arm
                ]

        # Input the second block
        outs = [self.feed_hierblock2(self.HBs2[i], outs[i],
                                 **kwargs) for i in range(len(self.HBs2))]

        ### Second fusion: left leg + right leg with left arm + right arm
        outs = self.feed_fusion(self.fuse_lay1, [outs[0], outs[1]], **kwargs) # (left leg + right leg) + (left arm + right arm)

        # Input the last block
        retval = self.feed_hierblock3(self.HB3, outs, **kwargs)

        n, j, s, h, w = sils.size()
        # Note that for all body parts j=1

        # Prepare input for visual summary
        sils_view = [
                        sils_left_leg.reshape(n * s, h, w),
                        sils_right_leg.reshape(n * s, h, w),
                        sils_left_arm.reshape(n * s, h, w),
                        sils_right_arm.reshape(n * s, h, w)
                    ]

        sils_view = torch.stack(sils_view, dim=2) # Add after h dim
        sils_view = sils_view.view(n*s, 1, h, -1)

        if 'visual_summary' not in retval:
            retval['visual_summary'] = {
                                        'image/sils': sils_view
                                       }

        return retval

    # Blocks and fusion forward definition
    def feed_hierblock1(self, block, x, **kwargs):
        return block(x)

    def feed_hierblock2(self, block, x, **kwargs):
        return block(x)

    def feed_hierblock3(self, block, x, **kwargs):
        return block(x)

    def feed_fusion(self, lay, inpts, **kwargs):
        outs = torch.stack(inpts, dim=0)

        if isinstance(lay, nn.Sequential):
            outs = lay(outs)
        else:
            outs = lay(outs, dim=0)

        return outs

from .gaitgl import GLConv, GeMHPP

class HiPoGaitGaitGL(MetaHiPoGait):
    """
        Definition of HiPo-Gait using GaitGL as backbone
    """

    def build_hierblock1(self, **kwargs):

        # Input parameters
        in_channels = 1
        in_c = kwargs['model_cfg']['channels']

        # Definition of input Conv3D
        conv3d = nn.Sequential(
            BasicConv3d(in_channels, in_c[0], kernel_size=(3, 3, 3),
                        stride=(1, 1, 1), padding=(1, 1, 1)),
            LayerChannelNorm(normalized_shape=in_c[0],
                             channel_dim=1, num_dims=5),
            nn.LeakyReLU(inplace=True)
        )

        # Define Conv3D output channel
        if not self.out_channel_block[0]:
            self.out_channel_block[0] = in_c[0]

        return conv3d

    def build_hierblock2(self, **kwargs):

        # Input parameters
        in_c = kwargs['model_cfg']['channels']

        # Definition of LTA layer
        LTA = nn.Sequential(
            BasicConv3d(in_c[0], in_c[0], kernel_size=(3, 1, 1),
                        stride=(3, 1, 1), padding=(0, 0, 0)),
            LayerChannelNorm(normalized_shape=in_c[0],
                             channel_dim=1, num_dims=5),
            nn.LeakyReLU(inplace=True)
        )

        # Define LTA output channel
        if not self.out_channel_block[1]:
            self.out_channel_block[1] = in_c[0]

        return LTA

    def build_hierblock3(self, **kwargs):

        # Input parameters
        in_c = kwargs['model_cfg']['channels']
        class_num = kwargs['model_cfg']['class_num']

        layers = {}

        # Define a GLConvA0 layer
        layers['GLConvA0'] = GLConv(in_channels=in_c[0],
                                    out_channels=in_c[1],
                                    halving=3,
                                    fm_sign=False,
                                    kernel_size=(3, 3, 3),
                                    stride=(1, 1, 1),
                                    padding=(1, 1, 1),
                                    normalize=True)

        ## MaxPool0
        layers['MaxPool0'] = nn.MaxPool3d(
            kernel_size=(1, 2, 2), stride=(1, 2, 2))

        ## GLConvA1
        layers['GLConvA1'] = GLConv(in_channels=in_c[1],
                                    out_channels=in_c[2],
                                    halving=3,
                                    fm_sign=False,
                                    kernel_size=(3, 3, 3),
                                    stride=(1, 1, 1),
                                    padding=(1, 1, 1),
                                    normalize=True)

        ## GLConvB2
        layers['GLConvB2'] = GLConv(in_channels=in_c[2],
                                    out_channels=in_c[2],
                                    halving=3,
                                    fm_sign=True,
                                    kernel_size=(3, 3, 3),
                                    stride=(1, 1, 1),
                                    padding=(1, 1, 1),
                                    normalize=True)

        layers['TP'] = PackSequenceWrapper(torch.max)
        layers['HPP'] = GeMHPP()

        layers['Head0'] = SeparateFCs(64, in_c[-1], in_c[-1])

        if 'SeparateBNNecks' in kwargs.keys():
            layers['BNNecks'] = SeparateBNNecks(**kwargs['SeparateBNNecks'])
            self.Bn_head = False
        else:
            layers['Bn'] = nn.BatchNorm1d(in_c[-1])
            layers['Head1'] = SeparateFCs(64, in_c[-1], class_num)
            self.Bn_head = True

        # Define output channel
        if not self.out_channel_block[2]:
            self.out_channel_block[2] = in_c[-1]

        return nn.ModuleDict(layers)

    def feed_hierblock1(self, block, x, **kwargs):

        labs = kwargs['labs']

        if not self.training and len(labs) != 1:
            raise ValueError(
                'The input size of each GPU must be 1 in testing mode, but got {}!'.format(len(labs)))

        # Ensure sequence length is divisible by 3
        s = x.size(2)
        if s < 3:
            repeat = 3 if s == 1 else 2
            x = x.repeat(1, 1, repeat, 1, 1)

        return super(HiPoGaitGaitGL, self).feed_hierblock1(block, x, **kwargs)

    def feed_hierblock3(self, block, x, **kwargs):

        seqL = kwargs['seqL']
        seqL = None if not self.training else seqL
        labs = kwargs['labs']

        outs = block['GLConvA0'](x)
        outs = block['MaxPool0'](outs)
        outs = block['GLConvA1'](outs)
        outs = block['GLConvB2'](outs)  # [n, c, s, h, w]

        outs = block['TP'](outs, seqL=seqL, options={"dim": 2})[0]  # [n, c, h, w]
        outs = block['HPP'](outs)  # [n, c, p]

        gait = block['Head0'](outs)  # [n, c, p]

        if self.Bn_head:  # Original GaitGL Head
            bnft = block['Bn'](gait)  # [n, c, p]
            logi = block['Head1'](bnft)  # [n, c, p]
            embed = bnft
        else:  # BNNechk as Head
            bnft, logi = block['BNNecks'](gait)  # [n, c, p]
            embed = gait

        retval = {
            'training_feat': {
                'triplet': {'embeddings': embed, 'labels': labs},
                'softmax': {'logits': logi, 'labels': labs}
            },
            'inference_feat': {
                'embeddings': embed
            }
        }

        return retval

from ..modules import SetBlockWrapper, HorizontalPoolingPyramid, conv1x1, conv3x3, BasicBlock2D, BasicBlockP3D, BasicBlock3D
from .deepgaitv2 import blocks_map

class HiPoGaitDeepGaitV2(MetaHiPoGait):
    """
        Definition of HiPo-Gait over the DeepGaitV2 architecture
    """
    def build_network(self, model_cfg):
        mode = model_cfg['Backbone']['mode']
        assert mode in blocks_map.keys()
        block = blocks_map[mode]

        layers      = model_cfg['Backbone']['layers']
        channels    = model_cfg['Backbone']['channels']
        self.inference_use_emb2 = model_cfg['use_emb2'] if 'use_emb2' in model_cfg else False

        if mode == '3d':
            strides = [
                [1, 1],
                [1, 2, 2],
                [1, 2, 2],
                [1, 1, 1]
            ]
        else:
            strides = [
                [1, 1],
                [2, 2],
                [2, 2],
                [1, 1]
            ]

        self.inplanes = [channels[0]] * (5 + 1) # Create a separate inplane for each layer (+1 to prevent index out)

        self.HPP = HorizontalPoolingPyramid(bin_num=[16])

        # Note all the conf. parameter required to build the DeepGaitV2 architecture
        self.conf_parameters = {'block': block,
                                'mode': mode,
                                'layers': layers,
                                'channels': channels,
                                'strides': strides}

        # Call the HiPo-Gait general builder
        super(HiPoGaitDeepGaitV2, self).build_network(model_cfg)

    def build_hierblock1(self, **kwargs):

        # Define layer 0 output channel
        if not self.out_channel_block[0]:
            self.out_channel_block[0] = self.inplanes[0]

        # Definition of layer 0
        seq_layers = SetBlockWrapper(nn.Sequential(
                            conv3x3(1, self.inplanes[0], 1),
                            nn.BatchNorm2d(self.inplanes[0]),
                            nn.ReLU(inplace=True))
        )

        return seq_layers

    def build_hierblock2(self, **kwargs):
        # Check if input # of channels should be duplicated
        num_channels = self.conf_parameters['channels'][0]

        # Define layer 1 output channel
        if not self.out_channel_block[1]:
            self.out_channel_block[1] = num_channels

        # Definition of layer 1
        return SetBlockWrapper(self.make_layer(BasicBlock2D,
                                               num_channels,
                                               self.conf_parameters['strides'][0],
                                               blocks_num=self.conf_parameters['layers'][0],
                                               mode=self.conf_parameters['mode'],
                                               layer_idx=1))

    def build_hierblock3(self, **kwargs):

        # Definition of layer 2, 3 & 4
        layer2 = self.make_layer(self.conf_parameters['block'],
                                 self.conf_parameters['channels'][1],
                                 self.conf_parameters['strides'][1],
                                 blocks_num=self.conf_parameters['layers'][1],
                                 mode=self.conf_parameters['mode'],
                                 layer_idx=2)
        layer3 = self.make_layer(self.conf_parameters['block'],
                                 self.conf_parameters['channels'][2],
                                 self.conf_parameters['strides'][2],
                                 blocks_num=self.conf_parameters['layers'][2],
                                 mode=self.conf_parameters['mode'],
                                 layer_idx=3)
        layer4 = self.make_layer(self.conf_parameters['block'],
                                 self.conf_parameters['channels'][3],
                                 self.conf_parameters['strides'][3],
                                 blocks_num=self.conf_parameters['layers'][3],
                                 mode=self.conf_parameters['mode'],
                                 layer_idx=4)

        if self.conf_parameters['mode'] == '2d':
            layer2 = SetBlockWrapper(layer2)
            layer3 = SetBlockWrapper(layer3)
            layer4 = SetBlockWrapper(layer4)

        FCs = SeparateFCs(16, self.conf_parameters['channels'][3], self.conf_parameters['channels'][2])
        BNNecks = SeparateBNNecks(16, self.conf_parameters['channels'][2],
                                       class_num=kwargs['model_cfg']['SeparateBNNecks']['class_num'])

        TP = PackSequenceWrapper(torch.max)
        # HPP already defined in build_network

        # Define output channel
        if not self.out_channel_block[2]:
            self.out_channel_block[2] = self.conf_parameters['channels'][2]

        return nn.ModuleDict({'layer2': layer2, 'layer3': layer3, 'layer4': layer4, 'FCs': FCs, 'BNNecks':BNNecks, 'TP': TP})

    def make_layer(self, block, planes, stride, blocks_num, layer_idx, mode='2d'):

        if max(stride) > 1 or (self.inplanes[layer_idx]) != planes * block.expansion:
            if mode == '3d':
                downsample = nn.Sequential(nn.Conv3d(self.inplanes[layer_idx], planes * block.expansion, kernel_size=[1, 1, 1], stride=stride, padding=[0, 0, 0], bias=False), nn.BatchNorm3d(planes * block.expansion))
            elif mode == '2d':
                downsample = nn.Sequential(conv1x1(self.inplanes[layer_idx], planes * block.expansion, stride=stride), nn.BatchNorm2d(planes * block.expansion))
            elif mode == 'p3d':
                downsample = nn.Sequential(nn.Conv3d(self.inplanes[layer_idx], planes * block.expansion, kernel_size=[1, 1, 1], stride=[1, *stride], padding=[0, 0, 0], bias=False), nn.BatchNorm3d(planes * block.expansion))
            else:
                raise TypeError('xxx')
        else:
            downsample = lambda x: x

        layers = [block(self.inplanes[layer_idx], planes, stride=stride, downsample=downsample)]
        self.inplanes[layer_idx + 1] = planes * block.expansion
        s = [1, 1] if mode in ['2d', 'p3d'] else [1, 1, 1]
        for i in range(1, blocks_num):
            layers.append(
                    block(self.inplanes[layer_idx + 1], planes, stride=s)
            )
        return nn.Sequential(*layers)

    def feed_hierblock1(self, block, x, **kwargs):
        assert x.size(-1) in [44, 88]
        return super(HiPoGaitDeepGaitV2, self).feed_hierblock1(block, x, **kwargs)

    def feed_hierblock3(self, block, x, **kwargs):

        seqL = kwargs['seqL']
        labs = kwargs['labs']

        # Feed to the rest of architecture
        out2 = block['layer2'](x)
        out3 = block['layer3'](out2)
        out4 = block['layer4'](out3)  # [n, c, s, h, w]

        # Temporal Pooling, TP
        outs = block['TP'](out4, seqL, options={"dim": 2})[0]  # [n, c, h, w]

        # Horizontal Pooling Matching, HPM
        feat = self.HPP(outs)  # [n, c, p]

        embed_1 = block['FCs'](feat)  # [n, c, p]
        embed_2, logits = block['BNNecks'](embed_1)  # [n, c, p]

        if self.inference_use_emb2:
            embed = embed_2
        else:
            embed = embed_1

        retval = {
            'training_feat': {
                'triplet': {'embeddings': embed_1, 'labels': labs},
                'softmax': {'logits': logits, 'labels': labs}
            },
            'inference_feat': {
                'embeddings': embed
            }
        }

        return retval
