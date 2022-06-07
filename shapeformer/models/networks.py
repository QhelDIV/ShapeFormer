import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import numpy as np


###############################################################################
# Helper Functions
###############################################################################

class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(
            nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(
            nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = Identity
    else:
        raise NotImplementedError(
            'normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count -
                             opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(
            optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m._class__._name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError(
                    'initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


def build_sdfnetwork(input_dim=3, init_radius=0.):
    net = SDFModule(input_dim=input_dim)
    # for k, v in net.named_parameters():
    #     if 'weight' in k:
    #         std = np.sqrt(2) / np.sqrt(v.shape[0])
    #         nn.init.normal_(v, 0.0, std)
    #     if 'bias' in k:
    #         nn.init.constant_(v, 0)
    #     if k == 'l_out.weight':
    #         std = np.sqrt(np.pi) / np.sqrt(v.shape[1])
    #         nn.init.constant_(v, std)
    #     if k == 'l_out.bias':
    #         nn.init.constant_(v, -init_radius)

    return net

##############################################################################
# Classes
##############################################################################


class Dense(nn.Module):
    def __init__(self, in_channel,  out_channel, bn=False, activation='relu'):
        super().__init__()

        self.dense = nn.Linear(in_channel, out_channel)
        # nn.init.kaiming_normal_(self.dense.weight)
        # nn.init.kaiming_normal_(self.dense.bias)
        # self.dense.bias.data.fill_(50.)
        self.activation = activation
        self.use_bn = bn
        if bn == True:
            self.bn = nn.BatchNorm1d(out_channel)
        if self.activation is None:
            self.actfunc = Identity()
        elif self.activation == 'relu':
            self.actfunc = F.relu
        elif self.activation == 'sigmoid':
            self.actfunc = F.sigmoid
        else:
            raise NotImplementedError(
                'activation %s is not supported here' % self.activation)

    def forward(self, x):
        x = self.dense(x)
        if self.use_bn == True:
            x = self.bn(x)
        x = self.actfunc(x)
        return x


class MLP(nn.Module):
    def __init__(self, spec=[2, 128, 128, 1], endActivation=True, activation='relu', endActFunc='relu'):
        super().__init__()
        layers = []
        for i in range(len(spec)-2):
            layers.append(Dense(spec[i], spec[i+1], activation=activation))
        if endActivation == True:
            layers.append(Dense(spec[-2], spec[-1], activation=endActFunc))
        else:
            layers.append(Dense(spec[-2], spec[-1], activation=None))

        self.model = nn.Sequential(*layers)
        self.layers = layers

    def forward(self, x):
        ''' x: tensor (B,..., spec[0])'''
        shape = x.shape
        # send in (Bx...,spec[0]) and reshape the output
        x = self.model(x.view(-1, shape[-1])).view(shape[:-1]+(-1,))

        return x


class TensorModule(nn.Module):
    def __init__(self, shape, tensor=None):
        super(TensorModule, self)._init__()
        if tensor is not None:
            self.tensor = torch.nn.Parameter(tensor)
        else:
            self.tensor = torch.nn.Parameter(torch.randn(
                *shape, requires_grad=True)/torch.tensor(shape).sum())

    def forward(self, x=None):
        return self.tensor


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(
            x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class CSDFModule(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.latent_dim = latent_dim = 512
        self.l1 = nn.Conv1d(input_dim, latent_dim,  1)
        self.l2 = nn.Conv1d(latent_dim, latent_dim, 1)
        self.l3 = nn.Conv1d(latent_dim, latent_dim, 1)
        self.l4 = nn.Conv1d(latent_dim, latent_dim - input_dim, 1)
        self.l5 = nn.Conv1d(latent_dim, latent_dim, 1)
        self.l6 = nn.Conv1d(latent_dim, latent_dim, 1)
        self.l7 = nn.Conv1d(latent_dim, latent_dim, 1)
        self.l_out = nn.Conv1d(latent_dim, 1, 1)
        self.beta = 100
        self.bn_1 = nn.BatchNorm1d(latent_dim, affine=False)
        self.bn_2 = nn.BatchNorm1d(latent_dim, affine=True)
        self.bn_3 = nn.BatchNorm1d(latent_dim, affine=True)
        self.bn_4 = nn.BatchNorm1d(latent_dim, affine=True)
        self.bn_5 = nn.BatchNorm1d(latent_dim, affine=True)
        self.bn_6 = nn.BatchNorm1d(latent_dim, affine=True)

    def forward(self, x):
        x = x.transpose(1, 2)
        h = F.softplus(self.bn_1(self.l1(x)), beta=self.beta)
        h = F.softplus(self.l2(h), beta=self.beta)
        h = F.softplus(self.l3(h), beta=self.beta)
        h = F.softplus(self.l4(h), beta=self.beta)
        h = torch.cat((h, x), axis=1)
        h = F.softplus(self.l5(h), beta=self.beta)
        h = F.softplus(self.l6(h), beta=self.beta)
        h = F.softplus(self.l7(h), beta=self.beta)
        h = self.l_out(h)
        h = h.transpose(1, 2)
        return h


class SDFModule(nn.Module):
    def __init__(self, input_dim, latent_dim=512):
        super().__init__()
        self.latent_dim = latent_dim
        self.l1 = nn.Linear(input_dim, latent_dim)
        self.l2 = nn.Linear(latent_dim, latent_dim)
        self.l3 = nn.Linear(latent_dim, latent_dim)
        self.l4 = nn.Linear(latent_dim, latent_dim - input_dim)
        self.l5 = nn.Linear(latent_dim, latent_dim)
        self.l6 = nn.Linear(latent_dim, latent_dim)
        self.l7 = nn.Linear(latent_dim, latent_dim)
        self.l_out = nn.Linear(latent_dim, 1)
        self.beta = 100

    def forward(self, x):
        h = F.softplus(self.l1(x), beta=self.beta)
        h = F.softplus(self.l2(h), beta=self.beta)
        h = F.softplus(self.l3(h), beta=self.beta)
        h = F.softplus(self.l4(h), beta=self.beta)
        h = torch.cat((h, x), axis=-1)
        h = F.softplus(self.l5(h), beta=self.beta)
        h = F.softplus(self.l6(h), beta=self.beta)
        h = F.softplus(self.l7(h), beta=self.beta)
        h = self.l_out(h)
        return h


class SoftPlus(nn.Module):
    def __init__(self, beta=100):
        super().__init__()
        self.beta = beta

    def __call__(self, x):
        return F.softplus(x, beta=self.beta)


class ImplicitModule(nn.Module):
    def __init__(self, input_dim, output_dim, latent_dim=512, hidden_layers=7, activation='softplus', skip=True):
        ''' activation : ['relu','softplus', 'relu', 'sin'] '''
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_layers, self.skip = hidden_layers, skip

        self.l1 = nn.Linear(input_dim, latent_dim)
        self.l2 = nn.Linear(latent_dim, latent_dim)
        self.l3 = nn.Linear(latent_dim, latent_dim)
        self.l4 = nn.Linear(
            latent_dim, (latent_dim - input_dim) if skip else latent_dim)
        self.l5 = nn.Linear(latent_dim, latent_dim)
        self.l6 = nn.Linear(latent_dim, latent_dim)
        self.l7 = nn.Linear(latent_dim, latent_dim)
        self.l_out = nn.Linear(latent_dim, output_dim)

        acts = {'relu': F.relu,
                'softplus': SoftPlus(beta=100),
                'sin': torch.sin,
                }
        self.act = acts[activation]

    def forward(self, x):
        h = self.act(self.l1(x))
        h = self.act(self.l2(h))
        h = self.act(self.l3(h))
        h = self.act(self.l4(h))
        if self.skip:
            h = torch.cat((h, x), axis=-1)
        h = self.act(self.l5(h))
        h = self.act(self.l6(h))
        h = self.act(self.l7(h))
        h = self.l_out(h)
        return h


class SineLayer(nn.Module):
    # from the SIREN project
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.
    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a
    # hyperparameter.
    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features,
                                            1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                            np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))

    def forward_with_intermediate(self, input):
        # For visualization of activation distributions
        intermediate = self.omega_0 * self.linear(input)
        return torch.sin(intermediate), intermediate


class ImplicitModule2(nn.Module):
    def __init__(self, input_dim, output_dim, latent_dim=512, hidden_layers=7, activation='softplus', skip=True):
        ''' activation : ['relu','softplus', 'relu', 'sin'] '''
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_layers, self.skip = hidden_layers, skip

        for i in range(hidden_layers):
            self.net.append

        self.l1 = nn.Linear(input_dim, latent_dim)
        self.l2 = nn.Linear(latent_dim, latent_dim)
        self.l3 = nn.Linear(latent_dim, latent_dim)
        self.l4 = nn.Linear(
            latent_dim, (latent_dim - input_dim) if skip else latent_dim)
        self.l5 = nn.Linear(latent_dim, latent_dim)
        self.l6 = nn.Linear(latent_dim, latent_dim)
        self.l7 = nn.Linear(latent_dim, latent_dim)
        self.l_out = nn.Linear(latent_dim, output_dim)

        acts = {'relu': F.relu,
                'softplus': SoftPlus(beta=100),
                'sin': torch.sin,
                }
        self.act = acts[activation]

    def forward(self, x):
        h = self.act(self.l1(x))
        h = self.act(self.l2(h))
        h = self.act(self.l3(h))
        h = self.act(self.l4(h))
        if self.skip:
            h = torch.cat((h, x), axis=-1)
        h = self.act(self.l5(h))
        h = self.act(self.l6(h))
        h = self.act(self.l7(h))
        h = self.l_out(h)
        return h


# Pytorch implementation of (A)NP modules migrated from https://github.com/deepmind/neural-processes
def split_context_target(x, y, num_context, num_extra_target):
    num_points = x.shape[1]
    locations = np.random.choice(
        num_points,
        size=num_context + num_extra_target,
        replace=False)
    x_context = x[:, locations[:num_context], :]
    y_context = y[:, locations[:num_context], :]
    x_target = x[:, locations, :]
    y_target = y[:, locations, :]
    return {'context_x': x_context, 'context_y': y_context,
            'target_x': x_target,  'target_y': y_target}
# Attention Modules
# Note that module with no parameters will not cause any problem.


class AttentionModule(nn.Module):
    """The Attention module."""

    def __init__(self, input_dims, rep, hidden_spec, normalize=True):
        # , att_type, scale=1., num_heads=8):
        """Create attention module.
            Takes in context inputs, target inputs and
            representations of each context input/output pair
            to output an aggregated representation of the context data.
            Args:
                input_dims: dimensions of key, query and value, i.e. (d_k, d_v)
                rep: transformation to apply to contexts before computing attention. 
                    One of: ['identity','mlp'].
                hidden_spec: list of number of hidden units per layer of mlp.
                    Used only if rep == 'mlp'.
                att_type: type of attention. One of the following:
                    ['uniform','laplace','dot_product','multihead']
                scale: scale of attention.
                normalize: Boolean determining whether to:
                    1. apply softmax to weights so that they sum to 1 across context pts or
                    2. apply custom transformation to have weights in [0,1].
                num_heads: number of heads for multihead.
        """
        super().__init__()
        self.d_k, self.d_v = input_dims
        self.rep = rep
        self.hidden_spec = hidden_spec
        self.normalize = normalize
        if self.rep == 'identity':
            self.k_rep = Identity
            self.q_rep = Identity
        elif self.rep == 'mlp':
            # Pass through MLP
            self.k_rep = MLP([self.d_k]+self.hidden_spec, endActivation=False)
            self.q_rep = MLP([self.d_k]+self.hidden_spec, endActivation=False)
            self.d_k, self.d_v = self.hidden_spec[-1], self.hidden_spec[-1]
        else:
            raise NameError("'rep' not among ['identity','mlp']")

    def forward(self, x1, x2, v):
        """Apply attention to create aggregated representation of v.
            Args:
                x1: tensor of shape [B,n1,d_x].
                x2: tensor of shape [B,n2,d_x].
                v: tensor of shape [B,n1,d].
            Returns:
                tensor of shape [B,n2,d]
            Raises:
                NameError: The argument for rep/type was invalid.
        """
        k = self.k_rep(x1)
        q = self.q_rep(x2)
        rep = self.forward_attention(q, k, v)
        return rep


class UniformAttentionModule(AttentionModule):
    def __init__(self, input_dims, rep, hidden_spec, normalize=True):
        super().__init__(input_dims, rep, hidden_spec, normalize)

    def forward_attention(self, q, k, v):
        """Uniform attention. Equivalent to np.
        Args:
            q: queries. tensor of shape [B,m,d_k].
            v: values. tensor of shape [B,n,d_v].
        Returns:
            tensor of shape [B,m,d_v].
        """
        total_points = q.shape[1]
        rep = v.mean(axis=1, keepdims=True)   # [B,1,d_v]
        rep = rep.repeat([1, total_points, 1])
        return rep  # [B,total_points,d_v]


def onePtanh(x):
    return 1 + torch.tanh(x)


class LaplaceAttentionModule(AttentionModule):
    def __init__(self, input_dims, rep, hidden_spec, scale, normalize=True):
        super().__init__(input_dims, rep, hidden_spec, normalize)
        self.scale = scale

    def forward_attention(self, q, k, v):
        """Computes laplace exponential attention.

        Args:
            q: queries. tensor of shape [B,m,d_k].
            k: keys. tensor of shape [B,n,d_k].
            v: values. tensor of shape [B,n,d_v].
            scale: float that scales the L1 distance.
            normalize: Boolean that determines whether weights sum to 1.

        Returns:
            tensor of shape [B,m,d_v].
        """
        k.unsqueeze(1)  # [B,1,n,d_k]
        q.unsqueeze(2)  # [B,m,1,d_k]
        unnorm_weights = - torch.abs((k - q) / scale)  # [B,m,n,d_k]
        unnorm_weights = unnorm_weights.sum(axis=-1)  # [B,m,n]
        if self.normalize:
            weight_fn = torch.nn.Softmax(dim=-1)
        else:
            weight_fn = onePtanh
        weights = weight_fn(unnorm_weights)  # [B,m,n]
        rep = torch.einsum('bik,bkj->bij', weights, v)  # [B,m,d_v]
        return rep


class DotProductAttentionModule(AttentionModule):
    def __init__(self, input_dims, rep, hidden_spec, normalize=True):
        super().__init__(input_dims, rep, hidden_spec, normalize)
        self.scale = np.sqrt(self.d_k)

    def forward_attention(self, q, k, v):
        """Computes dot product attention.

        Args:
            q: queries. tensor of  shape [B,m,d_k].
            k: keys. tensor of shape [B,n,d_k].
            v: values. tensor of shape [B,n,d_v].
            normalize: Boolean that determines whether weights sum to 1.

        Returns:
            tensor of shape [B,m,d_v].
        """
        unnorm_weights = torch.einsum(
            'bjk,bik->bij', k, q) / self.scale  # [B,m,n]
        if self.normalize:
            weight_fn = torch.nn.Softmax(dim=-1)
        else:
            weight_fn = torch.sigmoid
        weights = weight_fn(unnorm_weights)  # [B,m,n]
        rep = torch.einsum('bik,bkj->bij', weights, v)  # [B,m,d_v]
        return rep


class NCLConv1d(nn.Module):
    def __init__(self, conv1d_module):
        super().__init__()
        self.module = conv1d_module

    def forward(self, x):
        return self.module(x.permute(0, 2, 1)).permute(0, 2, 1)


class MultiheadAttentionModule(DotProductAttentionModule):
    def __init__(self, input_dims, rep, hidden_spec, normalize=True, num_heads=8):
        super().__init__(input_dims, rep, hidden_spec, normalize)
        self.num_heads = num_heads
        self.d_head = self.d_v // self.num_heads

        self.Wq, self.Wk, self.Wv, self.Wo = [], [], [], []
        for h in range(self.num_heads):
            self.Wq.append(NCLConv1d(torch.nn.Conv1d(
                self.d_k, self.d_head, 1, bias=False)))
            self.Wk.append(NCLConv1d(torch.nn.Conv1d(
                self.d_k, self.d_head, 1, bias=False)))
            self.Wv.append(NCLConv1d(torch.nn.Conv1d(
                self.d_v, self.d_head, 1, bias=False)))
            self.Wo.append(NCLConv1d(torch.nn.Conv1d(
                self.d_head, self.d_v, 1, bias=False)))

            self.add_module('WqH%d' % h, self.Wq[-1])
            self.add_module('WkH%d' % h, self.Wk[-1])
            self.add_module('WvH%d' % h, self.Wv[-1])
            self.add_module('WoH%d' % h, self.Wo[-1])

    def forward_attention(self, q, k, v):
        """Computes multi-head attention.

        Args:
            q: queries. tensor of  shape [B,m,d_k].
            k: keys. tensor of shape [B,n,d_k].
            v: values. tensor of shape [B,n,d_v].
            num_heads: number of heads. Should divide d_v.

        Returns:
            tensor of shape [B,m,d_v].
        """
        rep = []
        for h in range(self.num_heads):
            o = super().forward_attention(
                self.Wq[h](q), self.Wq[h](k), self.Wq[h](v))
            rep.append(self.Wo[h](o))
        rep = torch.stack(rep, axis=0).sum(axis=0)
        return rep


def get_attention(dim_x, dim_hidden, attentionType='uniform', attentionRep='mlp'):
    input_dims = (dim_x, dim_hidden)  # dimension of key and value
    hidden_spec = [dim_hidden]*3
    rep = attentionRep
    if attentionType == 'uniform':
        att = UniformAttentionModule(
            input_dims, rep, hidden_spec, normalize=True)
    elif attentionType == 'laplace':
        att = LaplaceAttentionModule(
            input_dims, rep, hidden_spec, normalize=True, scale=1.)
    elif attentionType == 'dot_product':
        att = DotProductAttentionModule(
            input_dims, rep, hidden_spec, normalize=True)
    elif attentionType == 'multihead':
        att = MultiheadAttentionModule(
            input_dims, rep, hidden_spec, normalize=True, num_heads=8)
    else:
        raise NameError("'attentionType' not among ['uniform','laplace','dot_product'"
                        ",'multihead']")
    return att


class DeterministicEncoder(nn.Module):
    """The Deterministic Encoder."""

    def __init__(self, spec, attention):
        """(A)NP deterministic encoder.

        Args:
          spec: An iterable containing the output sizes of the encoding MLP.
          attention: The attention module.
        """
        super().__init__()
        self.spec = spec
        self.attention = attention
        self.mlp = MLP(spec, endActivation=False)

    def forward(self, context_x, context_y, target_x):
        """Encodes the inputs into one representation.

        Args:
          context_x: Tensor of shape [B,observations,d_x]. For this 1D regression
              task this corresponds to the x-values.
          context_y: Tensor of shape [B,observations,d_y]. For this 1D regression
              task this corresponds to the y-values.
          target_x: Tensor of shape [B,target_observations,d_x]. 
              For this 1D regression task this corresponds to the x-values.

        Returns:
            The encoded representation. Tensor of shape [B,target_observations,d]
        """

        # Concatenate x and y along the filter axes
        encoder_input = torch.cat([context_x, context_y], axis=-1)

        # Pass final axis through MLP
        hidden = self.mlp(encoder_input)

        # Apply attention
        hidden = self.attention(context_x, target_x, hidden)

        return hidden


class LatentEncoder(nn.Module):
    """The Latent Encoder."""

    def __init__(self, spec, dim_latent):
        """(A)NP latent encoder.

        Args:
          spec: An iterable containing the output sizes of the encoding MLP.
          dim_latent: The latent dimensionality.
        """
        super().__init__()

        self.spec = spec
        self.dim_latent = dim_latent
        self.input_mlp = MLP(spec, endActivation=False)
        self.mu_sigma_mlp = MLP(
            [spec[-1]]*2+[2*dim_latent], endActivation=False)

    def forward(self, x, y):
        """Encodes the inputs into one representation.

            Args:
            x: Tensor of shape [B,observations,d_x]. For this 1D regression
                task this corresponds to the x-values.
            y: Tensor of shape [B,observations,d_y]. For this 1D regression
                task this corresponds to the y-values.

            Returns:
            A normal distribution over tensors of shape [B, dim_latent]
        """

        # Concatenate x and y along the filter axes
        encoder_input = torch.cat([x, y], axis=-1)

        # Pass final axis through MLP
        hidden = self.input_mlp(encoder_input)
        # Aggregator: take the mean over all points
        # TODO: replace this with self-attention module
        hidden = hidden.mean(axis=1)  # (B,dim_hidden)

        # Have further MLP layers that map to the parameters of the Gaussian latent
        # First apply intermediate relu layer
        # TODO: check whether sperate mu/log_sigma into 2 branch will help training
        hidden = self.mu_sigma_mlp(hidden)
        # Then apply further linear layers to output latent mu and log sigma
        mu = hidden[..., 0:self.dim_latent]*.1
        log_sigma = hidden[..., self.dim_latent:]
        #mu, log_sigma = hidden.chunk(2, axis=-1)

        # Compute sigma
        #sigma = 0.05 + 0.95 * tf.sigmoid(log_sigma)
        # TODO: check what will happen if the sigma is unbounded
        sigma = 0.01 + 0.99 * torch.sigmoid(log_sigma)

        dist = torch.distributions.normal.Normal(loc=mu, scale=sigma)
        # Multivariate Gaussian
        # dist = torch.distributions.independent.Independent( \
        #        torch.distributions.normal.Normal(loc = mu, scale = sigma),
        #        reinterpreted_batch_ndims = 1)
        # dist = torch.distributions.normal.Normal(loc = mu, scale = sigma) # (B,dim_latent)
        return dist  # , mu, sigma


class Decoder(nn.Module):
    """The Decoder."""

    def __init__(self, spec_hidden, dim_in, dim_out, min_std=.01, activation='relu'):
        """(A)NP decoder.
            Args:
            spec: An iterable containing the output sizes of the decoder MLP 
                as defined in `basic.Linear`.
        """
        super().__init__()
        dim_hidden = spec_hidden = [-1]
        self.mlp = MLP(spec=[dim_in]+spec_hidden, endActivation=False)
        self.mu_mlp = MLP(spec=[dim_hidden]*3+[dim_out], endActivation=False)
        self.std_mlp = MLP(spec=[dim_hidden]*3+[dim_out], endActivation=False)
        self.min_std = min_std
        if sdf_net == 'relu':
            self.implicit_decoder = build_sdfnetwork(input_dim=dim_in)
            self.implicit_decoder = ImplicitModule(input_dim=dim_in,
                                                   output_dim=dim_out,
                                                   latent_dim=dim_hidden,
                                                   activation=activation,
                                                   skip=self.opt.skip,
                                                   positional_encoding=self.opt.positional_encoding)
        else:
            from shapeformer.models.siren.networks import SIREN
            self.implicit_decoder = SIREN(in_features=dim_in, hidden_features=self.opt.dim_hidden, hidden_layers=7,
                                          out_features=dim_out, outermost_linear=True,
                                          first_omega_0=30, hidden_omega_0=30.)

        def forward(self, representation, target_x):
            """Decodes the individual targets.
            Args:
            representation: The representation of the context for target predictions. 
                Tensor of shape [B,target_observations,?].
            target_x: The x locations for the target query.
                Tensor of shape [B,target_observations,d_x].

            Returns:
            dist: A multivariate Gaussian over the target points. A distribution over
                tensors of shape [B,target_observations,d_y].
            mu: The mean of the multivariate Gaussian.
                Tensor of shape [B,target_observations,d_x].
            sigma: The standard deviation of the multivariate Gaussian.
                Tensor of shape [B,target_observations,d_x].
            """
            # concatenate target_x and representation
            data_in = torch.cat([target_x, representation], axis=-1)

            # Pass final axis through MLP
            hidden = self.mlp(data_in)

            # Get the mean an the variance
            mu = self.mu_mlp(hidden)
            log_sigma = self.std_mlp(hidden)

            # Bound the variance
            sigma = self.min_std + (1-self.min_std) * F.softplus(log_sigma)

            #mu = self.implicit_decoder(data_in)
            #mu = self.implicit_decoder(target_x)
            #mu = mu + target_x.norm(dim=-1, keepdim=True)
            mu = self.implicit_decoder(
                torch.cat([target_x, representation], axis=-1))

            dist = torch.distributions.normal.Normal(loc=mu, scale=sigma)
            # Multivariate Gaussian
            # dist = torch.distributions.independent.Independent( \
            #        torch.distributions.normal.Normal(loc = mu, scale = sigma),
            #        reinterpreted_batch_ndims = 1)

            return dist  # , mu, sigma


class ImplicitDecoder(nn.Module):
    """The Decoder."""

    def __init__(self, dim_repr=0, dim_x=3, dim_y=1, dim_hidden=256, activation='relu', skip=False,
                 positional_encoding=False, PE_L=10, dist_type='Gaussian', min_std=.01):
        """(A)NP decoder.
            Args:
            spec: An iterable containing the output sizes of the decoder MLP 
                as defined in `basic.Linear`.
        """
        super().__init__()
        self.dim_x, self.dim_in, self.dim_out, self.dist_type, self.min_std = dim_x, dim_repr, dim_y, dist_type, min_std
        self.positional_encoding, self.PE_L = positional_encoding, PE_L
        if positional_encoding == True:
            self.dim_pe_x = 2*self.dim_x*self.PE_L
            self.dim_in += self.dim_pe_x
        else:
            self.dim_in += self.dim_x

        if dist_type == 'Gaussian':
            self.dim_y_param = self.dim_out * 2
        elif dist_type == 'Bernoulli':
            self.dim_y_param = self.dim_out
        else:
            raise ValueError(f"Invalid distribution type: {dist_type}")

        self.implicit_decoder = ImplicitModule(input_dim=self.dim_in,
                                               output_dim=self.dim_y_param,
                                               latent_dim=dim_hidden,
                                               activation=activation,
                                               skip=skip,
                                               )

    def forward(self, representation=None, target_x=None):
        """Decodes the individual targets.
        Args:
        representation: The representation of the context for target predictions. 
            Tensor of shape [B,target_observations,?].
        target_x: The x locations for the target query.
            Tensor of shape [B,target_observations,d_x].
        Returns:
        dist: A multivariate Bernoulli over the target points. A distribution over
            tensors of shape [B,target_observations,d_y].
        """
        # concatenate target_x and representation
        if self.positional_encoding == True:
            L = self.PE_L
            positional_encoding = torch.zeros(
                target_x.shape[0], target_x.shape[1], self.dim_pe_x).type_as(target_x)
            for i in range(L):
                fac = 2**i
                for j in range(self.dim_x):
                    positional_encoding[:, :, 2*self.dim_x*i+2 *
                                        j] = torch.sin(fac * np.pi * target_x[..., j])
                    positional_encoding[:, :, 2*self.dim_x*i+2*j +
                                        1] = torch.cos(fac * np.pi * target_x[..., j])
            target_x = positional_encoding

        if representation is None:
            data_in = target_x
        else:
            data_in = torch.cat([target_x, representation], axis=-1)
        decoded = self.implicit_decoder(data_in)
        if self.dist_type == 'Bernoulli':
            dist = torch.distributions.bernoulli.Bernoulli(logits=decoded)
        elif self.dist_type == 'Gaussian':
            dim_y = decoded.shape[-1]//2
            mu, log_sigma = decoded[..., :dim_y], decoded[..., dim_y:]
            # Bound the variance
            sigma = self.min_std + (1-self.min_std) * F.softplus(log_sigma)
            dist = torch.distributions.normal.Normal(loc=mu, scale=sigma)
        return dist


############### UNIT TESTING ###############

def test_ImplicitDecoder():
    opts, decoders, results = [], [], []
    opts.append(dict(dim_repr=128, dim_x=3, dim_y=1, activation='relu', skip=False,
                     positional_encoding=True, PE_L=10, dist_type='Gaussian'))
    ImplicitDecoder(**opts[-1])(torch.zeros((3, 10, 128)),
                                torch.zeros((3, 10, 3)))

    opts.append(dict(dim_repr=128, dim_x=3, dim_y=1, activation='relu', skip=False,
                     positional_encoding=True, PE_L=10, dist_type='Bernoulli'))
    ImplicitDecoder(**opts[-1])(torch.zeros((3, 10, 128)),
                                torch.zeros((3, 10, 3)))

    opts.append(dict(dim_repr=0, dim_x=30, dim_y=2, activation='relu', skip=False,
                     positional_encoding=True, PE_L=10, dist_type='Gaussian'))
    ImplicitDecoder(**opts[-1])(None, torch.zeros((3, 10, 30)))
    opts.append(dict(dim_repr=0, dim_x=30, dim_y=2, activation='relu', skip=False,
                     positional_encoding=True, PE_L=10, dist_type='Bernoulli'))
    ImplicitDecoder(**opts[-1])(None, torch.zeros((3, 10, 30)))

    opts.append(dict(dim_repr=0, dim_x=3, dim_y=100, activation='relu', skip=True,
                     positional_encoding=False, PE_L=10, dist_type='Bernoulli'))
    ImplicitDecoder(**opts[-1])(None, torch.zeros((3, 10, 3)))
    # print(decoders[2](target_x=target_x))
    # print(decoders[3](target_x=target_x))
    #print(results[0], results[0].loc, results[0].scale)
    #print(results[1], results[1].logits)
