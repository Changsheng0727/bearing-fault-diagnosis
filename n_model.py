"""Backward-compatible import shim for the legacy notebook code."""

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from bearing_fault_diagnosis.models import CNNResNetModel


class CNN_ResNet_model(CNNResNetModel):
    """Preserve the original class name used by the notebooks."""

    def __init__(self, label_num, num_b, data_shape=(1000, 2)):
        super().__init__(label_count=label_num, num_blocks=num_b, data_shape=data_shape)

    def res_net_block(self, input_data):
        return self._residual_block(input_data)

    def model_create(self, learning_rate):
        return self.build(learning_rate)
