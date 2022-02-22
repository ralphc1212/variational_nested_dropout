import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
from functools import reduce
import operator
import math

device = torch.device('cuda:0')

eps = 1e-8
SCALE_POW = 1.

import math

class Conv2dVND_eval(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, ard_init=-10,
                 NUM_DIV=32, FREEZE_PART=2, PI=.9,
                 thresh=3, ord_mask_ones_prop=1., weight_prob_fwd=False, times_prob=True):
        bias = False  # Goes to nan if bias = True
        super(Conv2dVND_eval, self).__init__(in_channels, out_channels, kernel_size, stride,
                                        padding, dilation, groups, bias)

        self.NUM_DIV = NUM_DIV
        self.FREEZE_PART = FREEZE_PART
        self.PI = PI

        self.bias = None
        self.thresh = thresh
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.ard_init = ard_init
        self.log_alp = Parameter(ard_init * torch.ones_like(self.weight))

        self.EVERY = int(out_channels / NUM_DIV)
        DIM_BETA = NUM_DIV - FREEZE_PART

        self.beta_r = nn.Parameter(torch.ones(DIM_BETA) * 4, requires_grad=True)
        self.pi = nn.Parameter(PI * torch.ones(DIM_BETA), requires_grad=False)

        self.ONE = nn.Parameter(torch.tensor([1.]), requires_grad=False)
        self.ZERO = nn.Parameter(torch.tensor([0.]), requires_grad=False)
        self.pz = nn.Parameter(torch.cat([self.ONE, torch.cumprod(self.pi, dim=0)])
                               * torch.cat([1 - self.pi, self.ONE]), requires_grad=False)

        self.weight_prob_fwd = weight_prob_fwd
        self.omop = ord_mask_ones_prop

        self.times_prob = times_prob

    @staticmethod
    def clip(tensor, to=8):
        """
        Shrink all tensor's values to range [-to,to]
        """
        return torch.clamp(tensor, -to, to)

    @staticmethod
    def clip_alp(tensor, lwrb=8):
        """
        Shrink all tensor's values to range [-to,to]
        """
        return torch.clamp(tensor, -lwrb, 0.)

    @staticmethod
    def clip_beta(tensor, to=5.):
        """
        Shrink all tensor's values to range [-to,to]
        """
        return torch.clamp(tensor, -to, to)

    def set_weight_prob_fwd(self, weight_prob_fwd):
        assert type(weight_prob_fwd) is bool
        self.weight_prob_fwd = weight_prob_fwd
        # print('set: ', weight_prob_fwd)

    def set_ord_mask_ones_prop(self, ord_mask_ones_prop):
        """
        Setting the masks for frozen weights (not affected by nested dropout)
        """
        assert ord_mask_ones_prop >= 0. and ord_mask_ones_prop <= 1.
        self.omop = ord_mask_ones_prop
        NEW_PART = int((self.NUM_DIV - self.FREEZE_PART) * ord_mask_ones_prop)
        ZERO_PART = int(self.NUM_DIV - self.FREEZE_PART - NEW_PART)

        self.mask = torch.cat([self.ONE.repeat(self.FREEZE_PART),
                               self.ONE.repeat(NEW_PART),
                               torch.tensor([0.]).to(device).repeat(ZERO_PART)]).repeat_interleave(self.EVERY).to(device)
        # print('mask set ..., prop: ', self.omop)

    def set_scaling(self, scaling):
        assert isinstance(scaling, bool)
        self.times_prob = scaling

    def forward(self, input):
        """
        Forward with all regularized connections and random activations (Beyesian mode).
        """
        beta = F.sigmoid(self.clip_beta(self.beta_r))

        if self.training == False and self.weight_prob_fwd == False:
            prob = torch.cat([self.ONE.repeat(self.FREEZE_PART), torch.cumprod(beta, dim=0)**SCALE_POW]).repeat_interleave(self.EVERY)
            fmap = F.conv2d(input, self.weights_clipped,
                     self.bias, self.stride,
                     self.padding, self.dilation, self.groups)
            return fmap * self.mask.view(1, -1, 1, 1).expand_as(fmap) * prob.view(1, -1, 1, 1).expand_as(fmap)


        eps = 1e-8
        W = self.weight
        zeros = torch.zeros_like(W)
        clip_mask = self.get_clip_mask()
        conved_mu = F.conv2d(input, W, self.bias, self.stride,
                             self.padding, self.dilation, self.groups)
        log_alpha = self.clip_alp(self.log_alp)

        input2 = input * input
        alpha_w2 = torch.exp(log_alpha) * W * W
        conved_si = F.conv2d(input2, alpha_w2, self.bias, self.stride,
                                              self.padding, self.dilation, self.groups)

        conved_si = torch.sqrt(eps + conved_si)

        conved = conved_mu + \
                 conved_si * torch.normal(torch.zeros_like(conved_mu), torch.ones_like(conved_mu))


        if self.training == False and self.weight_prob_fwd == True:

            prob = torch.cat([self.ONE.repeat(self.FREEZE_PART), torch.cumprod(beta, dim=0)**SCALE_POW]).repeat_interleave(self.EVERY)

        # If normalized by the probability 
            if self.times_prob:
                return conved * self.mask.view(1, -1, 1, 1).expand_as(conved) * prob.view(1, -1, 1, 1).expand_as(conved)
            else:
                return conved * self.mask.view(1, -1, 1, 1).expand_as(conved)

        # Implementation of variational nested dropout
        qz = torch.cat([self.ONE, torch.cumprod(beta, dim=0)]) * torch.cat([1 - beta, self.ONE])
        sample = F.gumbel_softmax(qz, tau=1, hard=True)

        sum_ = torch.cumsum(sample, dim=0)
        dif = sum_ - sample
        mask0 = dif[1:]
        mask1 = 1. - mask0

        mask1 = torch.cat([self.ONE.repeat(self.FREEZE_PART), mask1]).repeat_interleave(self.EVERY)

        mask1 = mask1.view(1,-1,1,1).expand_as(conved)
        conved *= mask1
        # print('here 3.')
        return conved

    @property
    def weights_clipped(self):
        clip_mask = self.get_clip_mask()
        return torch.where(clip_mask, torch.zeros_like(self.weight), self.weight)

    def get_clip_mask(self):
        log_alp = self.clip_alp(self.log_alp)
        return torch.ge(log_alp, self.thresh)

    def train(self, mode):
        self.training = mode
        super(Conv2dVND_eval, self).train(mode)

    def get_reg(self, **kwargs):
        """
        Get weights regularization (KL(q(w)||p(w)) approximation)
        """

        log_alp = self.clip_alp(self.log_alp)
        element_wise_kl = .5 * log_alp \
                   + 1.16145124 * torch.exp(log_alp) \
                   - 1.50204118 * torch.exp(log_alp) ** 2 \
                   + 0.58629921 * torch.exp(log_alp) ** 3

        sum_kl = element_wise_kl.sum(dim=(1,2,3))
        beta = F.sigmoid(self.clip_beta(self.beta_r))

        qz = torch.cat([self.ONE, torch.cumprod(beta, dim=0)]) * torch.cat([1 - beta, self.ONE])
        coef0 = torch.cumsum(qz, dim=0)[:-1]
        coef1 = torch.sum(qz) - coef0
        coef1 = torch.cat([self.ONE.repeat(self.FREEZE_PART), coef1]).repeat_interleave(self.EVERY)

        kl_w = coef1.dot(sum_kl)

        qz = torch.cat([self.ONE, torch.cumprod(beta, dim=0)]) * torch.cat([1 - beta, self.ONE])
        log_frac_qz_pz = torch.log(qz / self.pz)
        kl_z = qz.dot(log_frac_qz_pz)


        kl = - (kl_w - kl_z)
        return kl

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_channels, self.out_channels, self.bias is not None
        )

    def get_dropped_params_cnt(self):
        """
        Get number of dropped weights (greater than "thresh" parameter)

        :returns (number of dropped weights, number of all weight)
        """
        return self.get_clip_mask().sum().cpu().numpy()

    @property
    def log_alpha(self):
        eps = 1e-8
        return self.log_sigma2 - 2 * torch.log(torch.abs(self.weight) + eps)


def get_ard_reg_vodo(module, reg=0):
    """

    :param module: model to evaluate ard regularization for
    :param reg: auxilary cumulative variable for recursion
    :return: total regularization for module
    """
    if isinstance(module, LinearVND_eval) or isinstance(module, Conv2dVND_eval): return reg + module.get_reg()
    if hasattr(module, 'children'): return reg + sum([get_ard_reg_vodo(submodule) for submodule in module.children()])
    return reg


def _get_dropped_params_cnt_vodo(module, cnt=0):
    if hasattr(module, 'get_dropped_params_cnt'): return cnt + module.get_dropped_params_cnt()
    if hasattr(module, 'children'): return cnt + sum(
        [_get_dropped_params_cnt_vodo(submodule) for submodule in module.children()])
    return cnt


def _get_params_cnt_vodo(module, cnt=0):
    if any([isinstance(module, LinearVND_eval), isinstance(module, Conv2dVND_eval)]): return cnt + reduce(operator.mul,
                                                                                                module.weight.shape, 1)
    if hasattr(module, 'children'): return cnt + sum(
        [_get_params_cnt_vodo(submodule) for submodule in module.children()])
    return cnt + sum(p.numel() for p in module.parameters())


def get_dropped_params_ratio_vodo(model):
    return _get_dropped_params_cnt_vodo(model) * 1.0 / _get_params_cnt_vodo(model)

