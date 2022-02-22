import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
from functools import reduce
import operator
import math
torch.backends.cudnn.deterministic = True

eps = 1e-8

class Conv2dVND(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, ard_init=-1, thresh=3, beta_r_init=3., NUM_DIV=32, FREEZE_PART=2, PI=.9):
        bias = False  # Goes to nan if bias = True
        super(Conv2dVND, self).__init__(in_channels, out_channels, kernel_size, stride,
                                        padding, dilation, groups, bias)

        self.NUM_DIV = NUM_DIV
        self.FREEZE_PART = FREEZE_PART
        self.PI = PI

        self.bias = None
        self.thresh = thresh
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.ard_init = ard_init
        self.log_alp = Parameter(ard_init * torch.ones_like(self.weight), requires_grad=True)

        self.EVERY = int(out_channels / NUM_DIV)
        DIM_BETA = NUM_DIV - FREEZE_PART

        self.beta_r = nn.Parameter(torch.ones(DIM_BETA) * beta_r_init, requires_grad=True)
        self.pi = nn.Parameter(PI * torch.ones(DIM_BETA), requires_grad=False)

        self.ONE = nn.Parameter(torch.tensor([1.]), requires_grad=False)
        self.ZERO = nn.Parameter(torch.tensor([0.]), requires_grad=False)
        self.pz = nn.Parameter(torch.cat([self.ONE, torch.cumprod(self.pi, dim=0)])
                               * torch.cat([1 - self.pi, self.ONE]), requires_grad=False)

        self.tau = 2.
        self.reset_parameters()

    @staticmethod
    def clip_beta(tensor, to=5.):
        """
        Shrink all tensor's values to range [-to,to]
        """
        return torch.clamp(tensor, -to, to)

    def forward(self, input):

        if self.training == False:
            return F.conv2d(input, self.weights_clipped,
                            self.bias, self.stride,
                            self.padding, self.dilation, self.groups)

        eps = 1e-8
        W = self.weight
        zeros = torch.zeros_like(W)
        clip_mask = self.get_clip_mask()
        conved_mu = F.conv2d(input, W, self.bias, self.stride,
                             self.padding, self.dilation, self.groups)
        log_alpha = self.log_alp

        conved_si = torch.sqrt(eps + F.conv2d(input * input,
                                              torch.exp(log_alpha) * W * W, self.bias, self.stride,
                                              self.padding, self.dilation, self.groups))

        conved = conved_mu + \
                 conved_si * torch.normal(torch.zeros_like(conved_mu), torch.ones_like(conved_mu))


        beta = F.sigmoid(self.clip_beta(self.beta_r))

        qz = torch.cat([self.ONE, torch.cumprod(beta, dim=0)]) * torch.cat([1 - beta, self.ONE])
        sample = F.gumbel_softmax(qz, tau=self.tau, hard=True)
        sum_ = torch.cumsum(sample, dim=0)
        dif = sum_ - sample
        mask0 = dif[1:]
        mask1 = 1. - mask0

        mask1 = torch.cat([self.ONE.repeat(self.FREEZE_PART), mask1]).repeat_interleave(self.EVERY)
        mask1 = mask1.view(1,-1,1,1).expand_as(conved)
        conved *= mask1

        return conved

    def set_tau(self, tau):
        print('set tau: ', tau)
        self.tau = tau

    @property
    def weights_clipped(self):
        clip_mask = self.get_clip_mask()
        return torch.where(clip_mask, torch.zeros_like(self.weight), self.weight)

    def get_clip_mask(self):
        log_alp = self.log_alp
        return torch.ge(log_alp, self.thresh)

    def train(self, mode):
        self.training = mode
        super(Conv2dVND, self).train(mode)

    def get_reg(self, **kwargs):
        """
        Get weights regularization - KL(q(W,z)||p(W,z)) approximation in Eq.11
        """

        # param 1
        # k1 = 0.792
        # k2 = -0.4826
        # k3 = 0.3451

        k1 = 0.7294
        k2 = - 0.2041
        k3 = 0.3492
        k4 = 0.5387

        # param 2
        # k1 = 0.6134
        # k2 = 0.2026
        # k3 = 0.7126

        log_alp = self.log_alp

        w_kl = -.5 * torch.log(1+1./(torch.exp(log_alp))) \
                          + k1 * torch.exp(-math.exp(k4)*(k2 + k3 * log_alp)**2)

        w_kl = w_kl.mean(dim=(1,2,3))

        beta = F.sigmoid(self.clip_beta(self.beta_r))

        qz = torch.cat([self.ONE, torch.cumprod(beta, dim=0)]) * torch.cat([1 - beta, self.ONE])
        coef0 = torch.cumsum(qz, dim=0)[:-1]
        coef1 = torch.sum(qz) - coef0
        coef1 = torch.cat([self.ONE.repeat(self.FREEZE_PART), coef1]).repeat_interleave(self.EVERY)

        kl_w = coef1.dot(w_kl)

        qz = torch.cat([self.ONE, torch.cumprod(beta, dim=0)]) * torch.cat([1 - beta, self.ONE])
        log_frac_qz_pz = torch.log(eps + qz / self.pz)
        kl_z = qz.dot(log_frac_qz_pz)

        kl = kl_z - kl_w
        return kl

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

def get_ard_reg_vnd(module, reg=0):
    """

    :param module: model to evaluate ard regularization for
    :param reg: auxilary cumulative variable for recursion
    :return: total regularization for module
    """
    if isinstance(module, Conv2dVND): return reg + module.get_reg()
    if hasattr(module, 'children'): return reg + sum([get_ard_reg_vnd(submodule) for submodule in module.children()])
    return reg

