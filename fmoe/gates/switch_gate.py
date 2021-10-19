r"""
Balanced gate with Switch Transformer's policy (Google, 2021)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .naive_gate import NaiveGate
from .utils import limit_by_capacity


class SwitchGate(NaiveGate):
    r"""
    A switch gate implementation
    """

    def __init__(self, d_model, num_expert, world_size, topk=1,
            switch_eps=.1, capacity=(1.2, 2.4),gate_all_comm=True, inner_gpu_cnt=4,save=False,layer_idx=-1,loss_k=1):
        assert topk == 1, 'topk should be 1 in switch'
        super().__init__(d_model, num_expert, world_size, top_k=1)
        self.switch_eps = switch_eps
        self.capacity = capacity
        # self.k = loss_k
        self.k=0.2
        print("loss k is:",self.k)

    def forward(self, inp):
        r"""
        The switch firstly conduct softmax and then calculates the top-1
        """
        score = self.gate(inp)

        if self.training:
            # random uniform number from [1-eps, 1+eps]
            noise = torch.rand_like(score)
            noise = noise * 2 * self.switch_eps + 1.0 - self.switch_eps
            score += noise

        # fp32 softmax for numerical stability
        score = F.softmax(score.float(), dim=-1)

        top1_score, top1_idx = torch.topk(
            score, k=1, dim=-1, largest=True
        )  # [.. x top_k]
        top1_score = top1_score.to(dtype=inp.dtype)

        # cap_rate = self.capacity[0 if self.training else 1]
        # capacity = math.ceil(cap_rate * inp.shape[0])
        # _new_lec, _new_gec, top1_idx = limit_by_capacity(
        #         top1_idx, self.num_expert, self.world_size, capacity)

        # valid_idx = top1_idx[top1_idx > -1]
        valid_idx = top1_idx[top1_idx > -1]    # fix bug

        fraction_expert = torch.scatter_add(
                torch.zeros(self.tot_expert, device=valid_idx.device),
                0,
                valid_idx,
                torch.ones_like(valid_idx, dtype=torch.float),
            ) / valid_idx.numel()
        prob_expert = score.sum(dim=0) / valid_idx.numel()
        
        loss = (fraction_expert * prob_expert).sum() * self.tot_expert

        cap_rate = self.capacity[0 if self.training else 1]
        capacity = math.ceil(cap_rate * inp.shape[0])
        _new_lec, _new_gec, top1_idx = limit_by_capacity(
                top1_idx, self.num_expert, self.world_size, capacity)

        # if torch.distributed.get_rank()==0:
        #     print(loss,valid_idx.numel())
            # quit()
        # print(loss,valid_idx.numel())
        self.set_loss(loss*self.k)
        return top1_idx, top1_score 
