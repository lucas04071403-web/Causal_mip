from torch import nn
import torch
import torch.nn.functional as F
from typing_extensions import override

class PartialLinear(nn.Module):
    def __init__(self, original_linear: nn.Linear, trainable_cols: list[int]):
        super().__init__()
        self.original_linear = original_linear
        self.num_rows = original_linear.in_features
        self.num_columns = original_linear.out_features

        self.trainable_cols = sorted(trainable_cols)
        self.register_buffer(
            "trainable_cols_tensor",
            torch.tensor(self.trainable_cols, dtype=torch.long),
            persistent=False,
        )

        self.original_linear.requires_grad_(False)
        self.trainable_weight = nn.Parameter(original_linear.weight[self.trainable_cols, :].clone())
        if original_linear.bias is None:
            self.trainable_bias = None
        else:
            self.trainable_bias = nn.Parameter(original_linear.bias[self.trainable_cols].clone())

        self.is_weight_changed = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            self.is_weight_changed = True
            frozen_output = self.original_linear(x)
            trainable_output = F.linear(x, self.trainable_weight, self.trainable_bias)
            cols = self.trainable_cols_tensor.to(device=x.device)
            return frozen_output.index_copy(frozen_output.dim() - 1, cols, trainable_output)
        else:
            if self.is_weight_changed:
                self.merge_to_linear()
                self.is_weight_changed = False
            return self.original_linear(x)
    
    def merge_to_linear(self):
        """Merge the trainable and frozen parts back into the original linear layer."""
        with torch.no_grad():
            self.original_linear.weight[self.trainable_cols, :] = self.trainable_weight
            if self.original_linear.bias is not None and self.trainable_bias is not None:
                self.original_linear.bias[self.trainable_cols] = self.trainable_bias
        return self.original_linear

