# Copyright (c) 2023, Zikang Zhou. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Dict, Optional

import torch
from torchmetrics import Metric

from metrics.utils import topk
from metrics.utils import valid_filter


class minFDE(Metric):

    def __init__(self,
                 max_guesses: int = 6,
                 **kwargs) -> None:
        super(minFDE, self).__init__(**kwargs)
        self.add_state('sum', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('count', default=torch.tensor(0), dist_reduce_fx='sum')
        self.max_guesses = max_guesses

    def update(self,
               pred: torch.Tensor,
               target: torch.Tensor,
               prob: Optional[torch.Tensor] = None,
               valid_mask: Optional[torch.Tensor] = None,
               keep_invalid_final_step: bool = True,
               return_individual: bool = False) -> Optional[Dict[str, torch.Tensor]]:
        if prob is None:
            pred_topk = pred
        else:
            pred, target, prob, valid_mask, _ = valid_filter(pred, target, prob, valid_mask, None, keep_invalid_final_step)
            pred_topk, _ = topk(self.max_guesses, pred, prob)
        inds_last = (valid_mask * torch.arange(1, valid_mask.size(-1) + 1, device=self.device)).argmax(dim=-1)
        per_agent = torch.norm(
            pred_topk[torch.arange(pred.size(0)), :, inds_last] -
            target[torch.arange(pred.size(0)), inds_last].unsqueeze(-2),
            p=2,
            dim=-1,
        ).min(dim=-1)[0]
        batch_sum = per_agent.sum()
        batch_count = pred.size(0)
        self.sum += batch_sum
        self.count += pred.size(0)

        if return_individual:
            return {
                'per_agent': per_agent,
                'batch_sum': batch_sum,
                'batch_count': torch.tensor(batch_count, device=pred.device, dtype=per_agent.dtype),
                'batch_mean': batch_sum / max(batch_count, 1),
            }
        return None

    def compute(self) -> torch.Tensor:
        return self.sum / self.count
