from typing import Optional

import torch
from torchmetrics import Metric

from metrics.utils import topk
from metrics.utils import valid_filter

class Coverage(Metric):
    """Compute whether the GT label is included in the top max_guesses highest probability

    Args:
        max_guesses (_type_): number of guesses to consider for coverage calculation
    """
    def __init__(self,
                 max_guesses: int = 6,
                 **kwargs) -> None:
        super(Coverage, self).__init__(**kwargs)
        self.add_state('sum', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('count', default=torch.tensor(0), dist_reduce_fx='sum')
        self.max_guesses = max_guesses

    def update(self,
               pred: torch.Tensor, # (N, S) probability of each anchor from the MRF predictor
               target: torch.Tensor, # (N, S) GT anchor label (one-hot)
        ) -> None:
        
        GT_anchor_label = target.argmax(dim=-1) # (N,) get the GT anchor label index
        topk_prob, topk_index = pred.topk(self.max_guesses) # (N, max_guesses) get the indices of the top max_guesses predicted anchors
        matches = (topk_index == GT_anchor_label.unsqueeze(-1)).any(dim=-1) # (N,) check if the GT anchor label is included in the top max_guesses predicted anchors
        self.sum += matches.sum()
        self.count += matches.numel()


    def compute(self) -> torch.Tensor:
        return self.sum / self.count