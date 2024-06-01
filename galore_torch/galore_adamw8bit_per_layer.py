import wandb
from bitsandbytes.optim.optimizer import Optimizer2State

import torch
import time

from mem_utils import mem_status
import bitsandbytes as bnb
from galore_torch import GaLoreAdamW8bit
from peft_pretraining import training_utils


class AdamW8bitPerLayer(Optimizer2State):
    def __init__(self, model, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2, id_galore_params=[],
                 args=None, min_8bit_size=4096, percentile_clipping=100, block_wise=True, is_paged=False):
        super().__init__("adam", model.parameters(), lr, betas, eps, weight_decay, 8, args, min_8bit_size,
                         percentile_clipping,
                         block_wise, is_paged=is_paged)
        n_params = len([0 for x in model.parameters() if x.requires_grad])
        # print('My init', n_params)
        self.lr = lr
        self.optimizer_dict = dict()
        self.opt_time_record = dict()
        galore_counter = 0
        ngalore_counter = 0

        for p in model.parameters():
            if p.requires_grad:
                if id(p) in id_galore_params:
                    galore_counter += 1

                    self.optimizer_dict[p] = GaLoreAdamW8bit([{'params': [p], 'rank': args.rank,
                                                               'update_proj_gap': args.update_proj_gap * 2,
                                                               'scale': args.galore_scale, 'n':args.n,
                                                               'proj_type': args.proj_type}],
                                                             lr=args.lr, weight_decay=args.weight_decay)
                else:
                    ngalore_counter += 1
                    self.optimizer_dict[p] = bnb.optim.Adam8bit([p], lr=args.lr, weight_decay=args.weight_decay)

        # get scheduler dict
        self.scheduler_dict = dict()
        for e, p in enumerate(model.parameters()):
            # print('Setting sch...', e, '/', n_params)
            if p.requires_grad:
                self.scheduler_dict[p] = training_utils.get_scheculer(
                    optimizer=self.optimizer_dict[p],
                    scheduler_type=args.scheduler,
                    num_training_steps=args.num_training_steps * 2,
                    warmup_steps=args.warmup_steps * 2,
                    min_lr_ratio=args.min_lr_ratio,
                )
                # print('Setting scheduler', self.scheduler_dict[p])

        def optimizer_hook(p):
            # print('Hook track', args.track_time)
            # print('Hook scheduler', self.scheduler_dict[p])
            if p.grad is None:
                return

            if args.track_time:
                torch.cuda.synchronize()
            opt_p_time = time.time()
            self.optimizer_dict[p].step()
            self.opt_time_record[id(p)] = time.time() - opt_p_time
            if args.track_time:
                torch.cuda.synchronize()

            self.optimizer_dict[p].zero_grad()
            self.scheduler_dict[p].step()

        # Register the hook onto every parameter
        for e, p in enumerate(model.parameters()):
            # print('Setting Hook...', e, '/', n_params)
            if p.requires_grad:
                p.register_post_accumulate_grad_hook(optimizer_hook)

        # print('galore_counter', galore_counter)
        # print('ngalore_counter', ngalore_counter)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        if not self.initialized:
            self.check_overrides()
            self.to_gpu()  # needed for fairseq pure fp16 training
            self.initialized = True

        return 0

    def get_lr(self):
        if len(self.optimizer_dict) != 0:
            self.lr = list(self.optimizer_dict.values())[0].param_groups[0]["lr"]
        return self.lr

    @staticmethod
    def sum_dicts_list(dict_list):
        if not dict_list:
            return dict()

        result = dict()
        for key in dict_list[0].keys():
            result[key] = sum(d.get(key, 0) for d in dict_list)  # sum
        return result

    def mem_status(self):
        mem_d = []
        for p in self.optimizer_dict:
            mem_d.append(mem_status(self.optimizer_dict[p],
                                    enable_galore=True, is8bit=True,
                                    report=False)['optim_states'])
        sum_of_states = {'optim_states': self.sum_dicts_list(mem_d)}
        print(sum_of_states)
        wandb.config.update(sum_of_states)
        return mem_d
