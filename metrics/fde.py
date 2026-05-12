from typing import Optional

import torch
from torchmetrics import Metric

class FDE(Metric):

    def __init__(self,
                 **kwargs) -> None:
        super(FDE, self).__init__(**kwargs)
        self.add_state('sum', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('count', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self,
               pred: torch.Tensor,
               target: torch.Tensor,
               valid_mask: Optional[torch.Tensor] = None,
               ):

        ind_last = (valid_mask * torch.arange(1, valid_mask.size(-1) + 1, device=self.device)).argmax(dim=-1)
        self.sum += torch.norm(pred[torch.arange(pred.size(0)), ind_last] - target[torch.arange(pred.size(0)), ind_last],
                               p=2, dim=-1).sum()
        self.count += pred.size(0)

    def compute(self) -> torch.Tensor:
        return self.sum / self.count