from easydict import EasyDict

from .padnet import InitialTaskPredictionModule, MultiTaskDistillationModule
from .aspp import DeepLabHead
from .layers import SABlock


class InitialTaskPredictionModule(InitialTaskPredictionModule):
    """Adapter for mtlpt.models.padnet.InitialTaskPredictionModule"""
    def __init__(self, aux_tasks_out_ch, tasks, input_channels, intermediate_channels=256):
        p = EasyDict({'AUXILARY_TASKS': {'NUM_OUTPUT': aux_tasks_out_ch}})
        super().__init__(p, tasks, input_channels, intermediate_channels)


class DeepLabHead(DeepLabHead):
    """Adapter for mtlpt.models.aspp.DeepLabHead, unifies interfaces to task heads."""
    def forward(self, enc_out, x=None, exec_layer=[]):
        return super().forward(enc_out[-1])


__all__ = ['InitialTaskPredictionModule', 'MultiTaskDistillationModule', 'DeepLabHead', 'SABlock']