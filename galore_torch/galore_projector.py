import math
from typing import Optional, Union

import torch
import numpy as np


def sample_gaussian_sketch(m, d, device, dtype):
    r = round(m * d) if d < 1 else int(d)
    mat = torch.randn(m, r, device=device, dtype=dtype)
    mat *= (1 / np.sqrt(r))
    return mat


class GaLoreProjector:
    def __init__(self, rank, verbose=False, update_proj_gap=200, scale=1.0, proj_type='std'):
        self.rank = rank
        self.verbose = verbose
        self.update_proj_gap = update_proj_gap
        self.scale = scale
        self.ortho_matrix = None
        self.proj_type = proj_type
        self.seed = None

    def project(self, full_rank_grad, iter):
        # change seed every gap iterations, and set it
        if (self.seed is None) or iter % self.update_proj_gap == 0:
            self.seed = torch.seed()

        torch.manual_seed(self.seed)

        if self.proj_type == 'std':
            if full_rank_grad.shape[0] >= full_rank_grad.shape[1]:
                if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
                    self.ortho_matrix = self.get_orthogonal_matrix(full_rank_grad, self.rank, type='right')
                low_rank_grad = torch.matmul(full_rank_grad, self.ortho_matrix.t())
            else:
                if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
                    self.ortho_matrix = self.get_orthogonal_matrix(full_rank_grad, self.rank, type='left')
                low_rank_grad = torch.matmul(self.ortho_matrix.t(), full_rank_grad)
        elif self.proj_type == 'reverse_std':
            if full_rank_grad.shape[0] >= full_rank_grad.shape[1]:
                if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
                    self.ortho_matrix = self.get_orthogonal_matrix(full_rank_grad, self.rank, type='left')
                low_rank_grad = torch.matmul(self.ortho_matrix.t(), full_rank_grad)
            else:
                if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
                    self.ortho_matrix = self.get_orthogonal_matrix(full_rank_grad, self.rank, type='right')
                low_rank_grad = torch.matmul(full_rank_grad, self.ortho_matrix.t())
        elif self.proj_type == 'right':
            if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
                self.ortho_matrix = self.get_orthogonal_matrix(full_rank_grad, self.rank, type='right')
            low_rank_grad = torch.matmul(full_rank_grad, self.ortho_matrix.t())
        elif self.proj_type == 'left':
            if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
                self.ortho_matrix = self.get_orthogonal_matrix(full_rank_grad, self.rank, type='left')
            low_rank_grad = torch.matmul(self.ortho_matrix.t(), full_rank_grad)
        elif self.proj_type == 'full':
            if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
                self.ortho_matrix = self.get_orthogonal_matrix(full_rank_grad, self.rank, type='full')
            low_rank_grad = torch.matmul(self.ortho_matrix[0].t(), full_rank_grad) @ self.ortho_matrix[1].t()
        elif self.proj_type == 'gaussian':
            if full_rank_grad.shape[0] >= full_rank_grad.shape[1]:
                self.ortho_matrix = sample_gaussian_sketch(full_rank_grad.shape[1], self.rank,
                                                           device=full_rank_grad.device,
                                                           dtype=full_rank_grad.dtype)
                low_rank_grad = torch.matmul(full_rank_grad, self.ortho_matrix)
            else:
                self.ortho_matrix = sample_gaussian_sketch(full_rank_grad.shape[0], self.rank,
                                                           device=full_rank_grad.device,
                                                           dtype=full_rank_grad.dtype)
                low_rank_grad = torch.matmul(self.ortho_matrix.t(), full_rank_grad)
        else:
            raise ValueError('Unsupported proj type', self.proj_type)

        # reset seed randomly
        torch.seed()
        return low_rank_grad

    def project_back(self, low_rank_grad):

        if self.proj_type == 'std':
            if low_rank_grad.shape[0] >= low_rank_grad.shape[1]:
                full_rank_grad = torch.matmul(low_rank_grad, self.ortho_matrix)
            else:
                full_rank_grad = torch.matmul(self.ortho_matrix, low_rank_grad)
        elif self.proj_type == 'reverse_std':
            if low_rank_grad.shape[0] <= low_rank_grad.shape[1]:  # note this is different from std
                full_rank_grad = torch.matmul(self.ortho_matrix, low_rank_grad)
            else:
                full_rank_grad = torch.matmul(low_rank_grad, self.ortho_matrix)
        elif self.proj_type == 'right':
            full_rank_grad = torch.matmul(low_rank_grad, self.ortho_matrix)
        elif self.proj_type == 'left':
            full_rank_grad = torch.matmul(self.ortho_matrix, low_rank_grad)
        elif self.proj_type == 'full':
            full_rank_grad = torch.matmul(self.ortho_matrix[0], low_rank_grad) @ self.ortho_matrix[1]
        elif self.proj_type in ['gaussian', 'gram', 'uniform', "JL"]:
            if low_rank_grad.shape[0] >= low_rank_grad.shape[1]:
                full_rank_grad = torch.matmul(low_rank_grad, self.ortho_matrix.t())
            else:
                full_rank_grad = torch.matmul(self.ortho_matrix, low_rank_grad)
            del self.ortho_matrix
        return full_rank_grad * self.scale

    # svd decomposition
    def get_orthogonal_matrix(self, weights, rank, type):
        rank = int(rank)
        module_params = weights

        if module_params.data.dtype != torch.float:
            float_data = False
            original_type = module_params.data.dtype
            original_device = module_params.data.device
            matrix = module_params.data.float()
        else:
            float_data = True
            matrix = module_params.data

        U, s, Vh = torch.linalg.svd(matrix, full_matrices=False)
        # print('matrix shape:', matrix.shape)
        # U, s, Vh = torch.linalg.svd(torch.randn_like(matrix), full_matrices=False)

        # make the smaller matrix always to be orthogonal matrix
        if type == 'right':
            A = U[:, :rank] @ torch.diag(s[:rank])
            B = Vh[:rank, :]

            if not float_data:
                B = B.to(original_device).type(original_type)
            return B
        elif type == 'left':
            A = U[:, :rank]
            B = torch.diag(s[:rank]) @ Vh[:rank, :]
            if not float_data:
                A = A.to(original_device).type(original_type)
            return A
        elif type == 'full':
            A = U[:, :rank]
            B = Vh[:rank, :]
            if not float_data:
                A = A.to(original_device).type(original_type)
                B = B.to(original_device).type(original_type)
            return [A, B]
        else:
            raise ValueError('type should be left, right or full')

    def numel(self):
        if hasattr(self, 'ortho_matrix') and self.ortho_matrix is not None:
            ortho_mat_list = self.ortho_matrix if isinstance(self.ortho_matrix, list) else [self.ortho_matrix]
            return sum([x.numel() for x in ortho_mat_list])
        else:
            return 0
