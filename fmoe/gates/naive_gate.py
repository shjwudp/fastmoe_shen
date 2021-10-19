r"""
Naive gate
"""
from .base_gate import BaseGate

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import os.path as osp
import errno
# from megatron import print_rank_0


def mkdir_if_missing(dirname):
    """Creates dirname if it is missing."""
    if not osp.exists(dirname):
        try:
            os.makedirs(dirname)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

class NaiveGate(BaseGate):
    r"""
    A naive gate implementation that defines the standard behavior of the gate
    which determines which experts the tokens are going to.
    Both the indicies and the score, or confidence, are output to the parent
    module.
    The load-balance strategies are also designed to be implemented within the
    `Gate` module.
    """

    def __init__(self, d_model, num_expert, world_size, top_k=2, gate_all_comm=True, inner_gpu_cnt=4,save=False,layer_idx=-1):
        super().__init__(num_expert, world_size)
        self.gate = nn.Linear(d_model, self.tot_expert)
        self.top_k = top_k
        self.inner_gpu = False
        self.gpu_per_machine = 8
        self.gate_all_comm = gate_all_comm
        self.inner_gpu_cnt = inner_gpu_cnt
        save=False
        self.save = save
        self.layer_idx = layer_idx
        if torch.distributed.get_rank()==0:
            print("layer_idx:"+str(layer_idx))
    
    
    def mask_other_node_expert(self,inp):
        
        
        # if self.inner_gpu:
        #     begin_idx = self.node_idx * self.num_expert
        #     end_idx = begin_idx+self.num_expert
        # else:
        #     machine_id = self.node_idx // self.gpu_per_machine
        #     begin_idx =  machine_id  * self.gpu_per_machine*self.num_expert
        #     end_idx =  (machine_id+1)  * self.gpu_per_machine*self.num_expert

        # machine_id = self.node_idx // self.gpu_per_machine
        # begin_idx =  machine_id  * self.gpu_per_machine*self.num_expert
        # end_idx =  (machine_id+1)  * self.gpu_per_machine*self.num_expert

        cluster_id =  self.node_idx // self.inner_gpu_cnt
        begin_idx = cluster_id*self.inner_gpu_cnt*self.num_expert
        end_idx =  (cluster_id+1)  * self.inner_gpu_cnt*self.num_expert
        inp[:,0:begin_idx] = -10000
        inp[:,end_idx:] = -10000
        # from megatron import print_rank_0
        # print_rank_0("begin_end_idx:"+str(begin_idx)+"_"+str(end_idx))
        # if self.node_idx==0:
        #     print(inp.shape,self.num_expert,self.world_size,begin_idx,end_idx)
            # print()

        return inp

        

    def forward(self, inp, return_all_scores=False):
        r"""
        The naive implementation simply calculates the top-k of a linear layer's
        output.
        """
        # if torch.distributed.get_rank()==0:
        #     print(self.gate)
        #     print(inp.shape,self.tot_expert)
        #     quit()
        gate = self.gate(inp)
        if not self.gate_all_comm:
            gate = self.mask_other_node_expert(gate)
        # from megatron import print_rank_0
        # print_rank_0("one forward"+str(self.gate_all_comm))
        gate_top_k_val, gate_top_k_idx = torch.topk(
            gate, k=self.top_k, dim=-1, largest=True, sorted=False
        )  # [.. x top_k]
        gate_top_k_val = gate_top_k_val.view(-1, self.top_k)
        # if torch.distributed.get_rank()==0:
        #     print("naive gate:",gate_top_k_val.shape,gate_top_k_val[8758],gate_top_k_idx[8758])
            # print()

        # (BxL) x 1 x top_k
        gate_score = F.softmax(gate_top_k_val, dim=-1)

        # if return_all_scores:
        #     return gate_top_k_idx, gate_top_k_val, gate
        # return gate_top_k_idx, gate_top_k_val
        if self.save and torch.distributed.get_rank()==2 and self.layer_idx!=-2:
            import time
            dir_name = osp.join("/hetu_group/shendong/code/Megatron-LM_v2.2/cpts/aba_study/83e_raw_model_gshare_gate_bt512_kloss_from_100k_origin/iter_0250000_test/tensor/layer_idx_"+str(self.layer_idx))
            mkdir_if_missing(dir_name)
            # print("tok_idx:",gate_top_k_idx)
            gate_top_k_idx_npy = gate_top_k_idx.detach().cpu().numpy()
            torch.save(gate_top_k_idx,dir_name+"/"+str(time.time()))

        # quit()
        if return_all_scores:
            return gate_top_k_idx, gate_score, gate
        return gate_top_k_idx, gate_score
