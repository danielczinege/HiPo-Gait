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
      foot_channels: [4, 5]   (left, right; default)
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
        # Input channels of the left and right foot maps (in this order)
        self.foot_channels = model_cfg.get('foot_channels', [4, 5])

        # HBs1[0] is the left leg and HBs1[1] the right leg; their first conv takes leg + foot, hence 2 input channels instead of 1
        for i in (0, 1):
            self.HBs1[i].forward_block[0] = conv3x3(2, self.inplanes[0], 1)

    def forward(self, inputs):
        # Same input layout handling as MetaHiPoGait.forward
        maps = inputs[0][0].transpose(1, 2)  # [n, s, j, h, w] -> [n, j, s, h, w]
        assert maps.size(1) > max(self.foot_channels), 'Input has no foot channels'

        # [left foot, right foot], each [n, 1, s, h, w], in the same order as HBs1[0] and HBs1[1]
        self.feet = [maps[:, c: c + 1] for c in self.foot_channels]

        return super().forward(inputs)

    def feed_hierblock1(self, block, x, **kwargs):
        # Append the same-side foot map to the leg map
        for i in (0, 1):
            if block is self.HBs1[i]:
                x = torch.cat([x, self.feet[i]], dim=1)

        return super().feed_hierblock1(block, x, **kwargs)
