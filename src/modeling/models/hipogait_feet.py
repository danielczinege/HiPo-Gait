"""
Condition C of the foot-keypoint experiments (DeepGaitV2 backbone only).

The same-side foot map is fed to each leg branch as a second input channel, so the first
convolution of the two leg branches takes two channels (leg, foot) instead of one. Everything
else is HiPo-Gait's DeepGaitV2 variant unchanged, and all weights are initialised as in HiPo-Gait.

Config, for pickles with the channels: 0 left arm, 1 left leg, 2 right arm, 3 right leg,
4 left foot, 5 right foot, 6 left leg+foot, 7 right leg+foot:

  A   model: HiPoGaitDeepGaitV2
  B   model: HiPoGaitDeepGaitV2
      body_part_channels: {left_arm: 0, left_leg: 6, right_arm: 2, right_leg: 7}
  C   model: HiPoGaitDeepGaitV2FeetChannel
      foot_channels: {left_leg: 4, right_leg: 5}   (default)
"""
import torch

from ..modules import conv3x3
from .hipogait import HiPoGaitDeepGaitV2


class HiPoGaitDeepGaitV2FeetChannel(HiPoGaitDeepGaitV2):
    """
        HiPo-Gait (DeepGaitV2) with the same-side foot map as a second input channel of each leg branch
    """

    def build_network(self, model_cfg):
        super().build_network(model_cfg)
        self.foot_channels = model_cfg.get('foot_channels', {'left_leg': 4, 'right_leg': 5})

        # HBs1[0] is the left leg and HBs1[1] the right leg; their first conv takes leg + foot
        for i in (0, 1):
            self.HBs1[i].forward_block[0] = conv3x3(2, self.inplanes[0], 1)

    def forward(self, inputs):
        # Same input layout handling as MetaHiPoGait.forward: [n, j, s, h, w]
        maps = inputs[0][0]
        maps = maps.unsqueeze(1) if maps.dim() == 4 else maps.transpose(1, 2)

        self.feet = [maps[:, self.foot_channels[side]: self.foot_channels[side] + 1]
                     for side in ('left_leg', 'right_leg')]

        return super().forward(inputs)

    def feed_hierblock1(self, block, x, **kwargs):
        # Append the same-side foot map to the leg map
        for i in (0, 1):
            if block is self.HBs1[i]:
                x = torch.cat([x, self.feet[i]], dim=1)

        return super().feed_hierblock1(block, x, **kwargs)
