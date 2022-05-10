import torch
import torch.nn as nn
import torch.nn.functional as F

class Quantizer(nn.Module):

    def __init__(self, vocab_size, n_embd, gamma=0.99, x_dim=3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, n_embd)
        self.embedding.weight.requires_grad = False

        self.n_embd = n_embd
        self.vocab_size = vocab_size
        self.gamma = gamma
        self.x_dim = x_dim

        self.register_buffer('N', torch.zeros(vocab_size))
        self.register_buffer('z_avg', self.embedding.weight.data.clone())
    def get_code(self, ind, bchw=True):
        quant_feat = self.embedding(ind)
        if bchw==True:
            if self.x_dim==1:
                quant_feat = quant_feat.permute(0, 2, 1).contiguous()
            if self.x_dim==2:
                quant_feat = quant_feat.permute(0, 3, 1, 2).contiguous()
            elif self.x_dim==3:
                quant_feat = quant_feat.permute(0, 4, 1, 2, 3).contiguous()
            else:
                raise ValueError("Only dim 2 and 3 supported.")
        return quant_feat
    def forward(self, grid_feat):
        if len(grid_feat.shape)==3:
            b, c, x1 = grid_feat.shape
            flat_inputs = grid_feat.permute(0, 2, 1).contiguous().view(-1, self.n_embd)
        elif len(grid_feat.shape)==4:
            b, c, x1, x2 = grid_feat.shape
            flat_inputs = grid_feat.permute(0, 2, 3, 1).contiguous().view(-1, self.n_embd)
        elif len(grid_feat.shape)==5:
            b, c, x1, x2, x3 = grid_feat.shape
            flat_inputs = grid_feat.permute(0, 2, 3, 4, 1).contiguous().view(-1, self.n_embd)
        weight = self.embedding.weight
        # (b*x1*x2,c) <- (b,x1,x2,c) <- (b,c,x1,x2)
        

        # (b*x1*x2, emb_size) <- (b*x1*x2,1) mat_add (b*x1*x2,emb_size) mat_add (1, emb_size)
            # (b*x1*x2, emb_size) <- (b*x1*x2,c) mm (c, emb_size)
        distances = (flat_inputs ** 2).sum(dim=1, keepdim=True) \
                    - 2 * torch.mm(flat_inputs, weight.t()) \
                    + (weight.t() ** 2).sum(dim=0, keepdim=True)
        # (b*x1*x2)
        encoding_indices = torch.max(-distances, dim=1)[1]
        # (b*x1*x2, emb_size)
        encode_onehot = F.one_hot(encoding_indices, self.vocab_size).type(flat_inputs.dtype)
        # (b,x1,x2)
        # (b,c,x1,x2) <- (b,x1,x2,c)
        if len(grid_feat.shape)==3:
            encoding_indices = encoding_indices.view(b, x1)
            quant_feat = self.embedding(encoding_indices).permute(0, 2, 1).contiguous()
        if len(grid_feat.shape)==4:
            encoding_indices = encoding_indices.view(b, x1, x2)
            quant_feat = self.embedding(encoding_indices).permute(0, 3, 1, 2).contiguous()
        elif len(grid_feat.shape)==5:
            encoding_indices = encoding_indices.view(b, x1, x2, x3)
            quant_feat = self.embedding(encoding_indices).permute(0, 4, 1, 2, 3).contiguous()

        # exponential weight average
        # computing cumulation of choices(onehot) and 
        if self.training:
            # inplace add_(other, alpha): +=other*alpha
            # (emb_size) <- (emb_size) * () mat_add ()*(emb_size)
            self.N.data.mul_(self.gamma).add_(1 - self.gamma, encode_onehot.sum(0))
            # (c, emb_size) <- (c, b*x1*x2) mm (b*x1*x2, emb_size)
            encode_sum = torch.mm(flat_inputs.t(), encode_onehot)
            # (emb_size, c) <- (emb_size, c)*() + ()*(emb_size, c)
            self.z_avg.data.mul_(self.gamma).add_(1 - self.gamma, encode_sum.t())
            # ()
            n = self.N.sum()
            # (emb_size) <- (emb_size) / ()
            weights = (self.N + 1e-7) / (n + self.vocab_size * 1e-7) * n
            # (emb_size, c) <- (emb_size, c) / (emb_size, 1)
            encode_normalized = self.z_avg / weights.unsqueeze(1)
            # (emb_size, c)
            self.embedding.weight.data.copy_(encode_normalized)

        quant_feat_st = (quant_feat - grid_feat).detach() + grid_feat
        quant_diff    = (grid_feat  - quant_feat.detach()).pow(2).mean()
        # e, e straight-through (used for loss back-prop), indices
        # (b,c,x1,x2), (b,c,x1,x2), (b,x1,x2)
        return quant_feat, quant_feat_st, encoding_indices, quant_diff
