"""
Condition C for the foot-keypoint chapter (DeepGaitV2 backbone).

Conditions A and B need no code at all. All three read the 8-channel pickles written by
fuse_heatmaps.py, and MetaHiPoGait.forward already slices its branches through
`body_part_channels`, which it reads from model_cfg:

  A   model: HiPoGaitDeepGaitV2
      (default body_part_channels: left_arm 0, left_leg 1, right_arm 2, right_leg 3)

  B   model: HiPoGaitDeepGaitV2
      body_part_channels: {left_arm: 0, left_leg: 6, right_arm: 2, right_leg: 7}
      Channels 6 and 7 are HiPo-Gait's own fusion of the six leg and foot keypoints as one
      group, so B is the released model, unmodified, reading the representation their
      pretreatment would produce with the feet in the leg group.

  C   model: HiPoGaitDeepGaitV2FeetChannel
      foot_channels: {left_leg: 4, right_leg: 5}

run_hipogait.sh stages only the channels a condition uses, already in this order (A: 0,1,2,3;
B: 0,6,2,7; C: 0,1,2,3,4,5). On that staged data all three run with the default
body_part_channels and foot_channels; the mappings above are what the same models need when they
read the full 8-channel pickles. The network sees the same values either way.

Condition C widens the first convolution of the two leg branches from one input channel to
two: conv3x3(1, 64) becomes conv3x3(2, 64), so +576 parameters per leg branch, +1,152 in
total. All weights are initialised the regular way (BaseModel.init_parameters, Xavier uniform).
The widening happens after the parent's initialisation: the leg slot keeps the weights it was
given there, and the new foot slot is drawn with Xavier uniform for the widened 2-channel
convolution, i.e. as if conv3x3(2, 64) had been built and initialised directly. These draws are
made on a forked random-number stream, so they do not advance the global one. With the same
seed (HiPo-Gait seeds every run the same way), every weight C shares with A and B therefore
starts at the same value as in their runs, and everything seeded after the model (e.g. the
batch sampling) is identical too.

The forward pass is not reimplemented. MetaHiPoGait.forward runs unchanged; only what the two
leg branches receive is altered, through the feed_hierblock1 hook the parent already provides.
That keeps C a single controlled difference from A rather than a parallel copy that could
drift, and it leaves every state-dict key identical to A's apart from the two widened weights.

Amplitude note. B's leg maps carry sqrt(3/6) of A's amplitude, because a group of n keypoints
is divided by sqrt(n). HB1 here is conv3x3 followed by BatchNorm2d, so a global input scale is
absorbed exactly and that difference cannot by itself move the result. This does not hold for
the GaitGL variant, whose first block has no normalisation, which is why this file is
DeepGaitV2 only.
"""
import torch
import torch.nn as nn

from .hipogait import HiPoGaitDeepGaitV2

LEG_BRANCHES = {0: 'left_leg', 1: 'right_leg'}   # MetaHiPoGait.forward's HBs1 order
DEFAULT_FOOT = {'left_leg': 4, 'right_leg': 5}


def _log(model, msg):
    mgr = getattr(model, 'msg_mgr', None)
    print(msg) if mgr is None else mgr.log_info(msg)


def _widen_first_conv(block: nn.Module, extra_in: int) -> nn.Module:
    """Give the first Conv2d/Conv3d in `block` `extra_in` more input channels.

    Existing weights move to the leading slots. The new trailing slots get the Xavier-uniform
    initialisation of BaseModel.init_parameters, computed for the widened shape. Draws from the
    global random-number stream; the caller decides whether to fork it.
    """
    for name, mod in block.named_modules():
        if not isinstance(mod, (nn.Conv2d, nn.Conv3d)):
            continue
        if name == '':
            raise RuntimeError('the block is itself a convolution; wrap it before widening')
        new = type(mod)(mod.in_channels + extra_in, mod.out_channels, mod.kernel_size,
                        stride=mod.stride, padding=mod.padding, dilation=mod.dilation,
                        groups=mod.groups, bias=mod.bias is not None)
        with torch.no_grad():
            nn.init.xavier_uniform_(new.weight)
            new.weight[:, :mod.in_channels] = mod.weight
            if mod.bias is not None:
                new.bias.copy_(mod.bias)
        parent_name, _, attr = name.rpartition('.')
        setattr(block.get_submodule(parent_name) if parent_name else block, attr, new)
        return new
    raise RuntimeError('No convolution found in block')


class HiPoGaitDeepGaitV2FeetChannel(HiPoGaitDeepGaitV2):
    """Condition C: a separate foot channel in each leg branch."""

    def build_network(self, model_cfg):
        super().build_network(model_cfg)                # exactly A's network, built the same way
        cfg = model_cfg.get('foot_channels', DEFAULT_FOOT)
        self.foot_channels = {k: int(v) for k, v in cfg.items()}

    def init_parameters(self):
        # BaseModel calls this once, right after build_network. Initialising first and widening
        # afterwards gives every weight C shares with A the same initial value for the same seed;
        # widening first would change the shapes the random draws see and shift every draw after
        # them. Both widenings share one forked stream, so the two leg branches get independent
        # foot weights while the global stream is left exactly where A's run leaves it.
        super().init_parameters()
        self._foot_convs, added = [], 0
        with torch.random.fork_rng(devices=[]):
            for i in LEG_BRANCHES:
                conv = _widen_first_conv(self.HBs1[i], extra_in=1)
                self._foot_convs.append(conv)
                added += conv.out_channels * conv.weight[0, 0].numel()
        _log(self, f'condition C: +{added} parameters, foot channels {self.foot_channels}')

    def feed_hierblock1(self, block, x, **kwargs):
        """The parent's hook. Leg branches get the same-side foot map appended.

        The foot map goes last, matching the trailing weights added by the widening.
        """
        for i, side in LEG_BRANCHES.items():
            if block is self.HBs1[i]:
                x = torch.cat([x, self._foot_maps[side]], dim=1)
                break
        # through the parent's hook, which also checks the input width
        return super().feed_hierblock1(block, x, **kwargs)

    def forward(self, inputs):
        sils = inputs[0][0]
        sils = sils.unsqueeze(1) if sils.dim() == 4 else sils.transpose(1, 2)
        assert sils.size(1) > max(self.foot_channels.values()), \
            f'expected >= {max(self.foot_channels.values()) + 1} channels, got {sils.size(1)}'
        self._foot_maps = {side: sils[:, c:c + 1].contiguous()
                           for side, c in self.foot_channels.items()}
        try:
            return super().forward(inputs)          # the parent forward, unchanged
        finally:
            self._foot_maps = None
