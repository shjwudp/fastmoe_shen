r"""
Supportive modules to conduct distributed training
"""
import torch
import torch.nn as nn
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
from .utils import get_torch_default_comm

import bagua.torch_api as bagua


class DistributedGroupedDataParallel(nn.Module):
    r"""
    A customized DDP module to support different all-reduce regions in the
    model.  The all-reduce region is defined as an attribution `dp_comm` in the
    weight object.
    The grads of the weights are identified to be reduced in different groups
    according to the weigths' `dp_comm` attribute.
    If it is set to `dp`, it will only be reduced across the data-parallel
    groups, which means that in the model parallel group, they are not
    synchronized.
    If it is set to `world`, the gradients is synchronized across all workers,
    regardless their model or data parallel group. This is extremely useful for
    shared layers like the gate.
    """

    def __init__(
        self,
        module,
        mp_group=None,
        dp_group=None,
        moe_group=None,
        world_group=None,
        auto_allreduce=False,
    ):
        assert not auto_allreduce, "Automatic all-reduce is not implemented yet"

        super().__init__()
        self.module = module

        self.comms = dict()
        if mp_group is not None:
            self.comms["mp"] = mp_group
        if dp_group is not None:
            self.comms["dp"] = dp_group
        else:
            self.comms["dp"] = get_torch_default_comm()
        if moe_group is not None:
            self.comms["moe"] = moe_group
        else:
            self.comms["moe"] = get_torch_default_comm()
        if world_group is None:
            self.comms["world"] = get_torch_default_comm()
        else:
            self.comms["world"] = world_group

        def allreduce_params(no_scale=False,
                reduce_after=False, fp32_allreduce=False):
            groups = dict()

            p_name = dict()
            for n, p in self.module.named_parameters():
                p_name[p] = n

            for p in self.module.parameters():
                if not p.requires_grad or p.grad is None:
                    continue
                if hasattr(p, "dp_comm"):
                    dp_comm = p.dp_comm
                else:
                    dp_comm = "dp"
                print("rank={}, dp_comm={}, name={}".format(bagua.get_rank(), dp_comm, p_name[p]), flush=True)
                group_key = (dp_comm, p.dtype)
                if group_key not in groups:
                    groups[group_key] = [p]
                else:
                    groups[group_key].append(p)
            for (dp_comm, dtype), group in groups.items():
                if dp_comm not in self.comms:
                    continue
                comm = self.comms[dp_comm]
                grads = [p.grad.data for p in group]
                coalesced = _flatten_dense_tensors(grads)
                if fp32_allreduce and dtype != torch.float32:
                    coalesced = coalesced.float()
                if not no_scale and not reduce_after:
                    coalesced /= comm.size()
                torch.distributed.all_reduce(coalesced, group=comm)
                torch.cuda.synchronize()
                if not no_scale and reduce_after:
                    coalesced /= comm.size()
                synced = _unflatten_dense_tensors(coalesced, grads)
                for g, s in zip(grads, synced):
                    g.copy_(s)

        self.allreduce_params = allreduce_params
        self._sync_params()

    def _sync_params(self):
        groups = dict()
        for p in self.module.parameters():
            if not p.requires_grad or p.grad is None:
                continue
            if hasattr(p, "dp_comm"):
                dp_comm = p.dp_comm
            else:
                dp_comm = "dp"
            group_key = (dp_comm, p.dtype)
            if group_key not in groups:
                groups[group_key] = [p]
            else:
                groups[group_key].append(p)
        for (dp_comm, _), group in groups.items():
            if dp_comm not in self.comms:
                continue
            comm = self.comms[dp_comm]
            datas = [p.data for p in group]
            coalesced = _flatten_dense_tensors(datas)
            torch.distributed.broadcast(coalesced, 0, group=comm)
            torch.cuda.synchronize()
            synced = _unflatten_dense_tensors(coalesced, datas)
            for d, s in zip(datas, synced):
                d.copy_(s)

    def forward(self, *args, **kwargs):
        r"""
        Directly call the module's forward function.
        """
        return self.module(*args, **kwargs)
