from bitsandbytes.optim.optimizer import Optimizer2State

import torch
import torch.nn.functional as F

from plotting import plot_spectrum

PLOT_SPECTRUM = False
from .galore_projector import GaLoreProjector


class AdamW8bit(Optimizer2State):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2, amsgrad=False, optim_bits=32,
                 args=None, min_8bit_size=4096, percentile_clipping=100, block_wise=True, is_paged=False):
        super().__init__("adam", params, lr, betas, eps, weight_decay, 8, args, min_8bit_size, percentile_clipping,
                         block_wise, is_paged=is_paged)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        overflows = []

        if not self.initialized:
            self.check_overrides()
            self.to_gpu()  # needed for fairseq pure fp16 training
            self.initialized = True
        thresholds = [0.001, 0.01, 0.05, 0.1]
        rank_histograms = dict()
        # if self.is_paged: self.page_mng.prefetch_all()
        for gindex, group in enumerate(self.param_groups):
            for pindex, p in enumerate(group["params"]):
                if p.grad is None:
                    continue
                state = self.state[p]

                if "step" not in state:
                    state["step"] = 0

                # GaLore Projection
                if "rank" in group:
                    if PLOT_SPECTRUM:
                        U, S, VT = torch.linalg.svd(p.grad)
                        for threshold in thresholds:
                            if threshold not in rank_histograms:
                                rank_histograms[threshold] = []
                            rank_histograms[threshold].append(torch.count_nonzero(S > threshold).item())
                    else:
                        if "projector" not in state:
                            state["projector"] = GaLoreProjector(group["rank"],
                                                                 update_proj_gap=group["update_proj_gap"],
                                                                 scale=group["scale"], proj_type=group["proj_type"],
                                                                 n=group["n"])

                        if 'weight_decay' in group and group['weight_decay'] > 0:
                            # ensure that the weight decay is not applied to the norm grad
                            group['weight_decay_saved'] = group['weight_decay']
                            group['weight_decay'] = 0
                        grad = state["projector"].project(p.grad, state["step"])

                        # suboptimal implementation
                        p.saved_data = p.data.clone()
                        p.data = grad.clone().to(p.data.dtype).to(p.data.device)
                        p.data.zero_()
                        p.grad = grad

                        if 'state1' not in state:
                            self.init_state(group, p, gindex, pindex)

                        self.prefetch_state(p)
                        self.update_step(group, p, gindex, pindex)
                        torch.cuda.synchronize()

                        # GaLore Projection Back
                        if "rank" in group:
                            p.data = p.saved_data.add_(state["projector"].project_back(p.data))

                            # apply weight decay
                            if 'weight_decay_saved' in group:
                                p.data.add_(p.data, alpha=-group['lr'] * group['weight_decay_saved'])
                                group['weight_decay'] = group['weight_decay_saved']
                                del group['weight_decay_saved']
        if PLOT_SPECTRUM:
            plot_spectrum(rank_histograms)
        if self.is_paged:
            # all paged operation are asynchronous, we need
            # to sync to make sure all tensors are in the right state
            torch.cuda.synchronize()

        return loss

    @staticmethod
    def compare_gradients(grad, new_grad):
        mse = F.mse_loss(grad, new_grad)
        cosine_similarity = F.cosine_similarity(grad, new_grad, dim=0)
        angle = torch.acos(cosine_similarity) * (180.0 / torch.pi)
        return {"grad_mse": mse,
                "grad_cosine_similarity": cosine_similarity,
                "grad_angle": angle}
