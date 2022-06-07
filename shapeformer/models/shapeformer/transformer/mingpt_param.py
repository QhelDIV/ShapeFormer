"""
taken from: https://github.com/karpathy/minGPT/
GPT model:
- the initial stem consists of a combination of token encoding and a positional encoding
- the meat of it is a uniform sequence of Transformer blocks
    - each Transformer is a sequential combination of a 1-hidden-layer MLP block and a self-attention block
    - all blocks feed into a central residual pathway similar to resnets
- the final decoder is a linear projection into a vanilla Softmax classifier
"""

import math
import logging

import torch
import torch.nn as nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)


class GPTConfig:
    """ base GPT config, params common to all GPT versions """
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    def __init__(self, vocab_size, block_size, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        for k,v in kwargs.items():
            setattr(self, k, v)


class GPT1Config(GPTConfig):
    """ GPT-1 like network roughly 125M params """
    n_layer = 12
    n_head = 12
    n_embd = 768


class GPT2Config(GPTConfig):
    """ GPT-2 like network roughly 1.5B params """
    # TODO


class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        mask = torch.tril(torch.ones(config.block_size,
                                     config.block_size))
        if hasattr(config, "n_unmasked"):
            mask[:config.n_unmasked, :config.n_unmasked] = 1
        self.register_buffer("mask", mask.view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head

    def forward(self, x, layer_past=None):
        B, T, C = x.size()
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y


class Block(nn.Module):
    """ an unassuming Transformer block """
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),  # nice
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    """  the full GPT language model, with a context size of block_size """
    def __init__(self, vocab_size, block_size, n_layer=12, n_head=8, n_embd=256,
                 embd_pdrop=0., resid_pdrop=0., attn_pdrop=0., n_unmasked=0):
        """GPT model

        Args:
            vocab_size (int): how many vocabularies
            block_size (int): The maximum sequence length
            n_layer (int, optional): Number of layers. Defaults to 12.
            n_head (int, optional): head num. Defaults to 8.
            n_embd (int, optional): embedding dimension. Defaults to 256.
            embd_pdrop (float, optional): ?. Defaults to 0..
            resid_pdrop (float, optional): ?. Defaults to 0..
            attn_pdrop (float, optional): ?. Defaults to 0..
            n_unmasked (int, optional): ?. Defaults to 0.
        """
        super().__init__()
        config = GPTConfig(vocab_size=vocab_size, block_size=block_size,
                           embd_pdrop=embd_pdrop, resid_pdrop=resid_pdrop, attn_pdrop=attn_pdrop,
                           n_layer=n_layer, n_head=n_head, n_embd=n_embd,
                           n_unmasked=n_unmasked, no_pos_emb=False)
        # input embedding stem
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.drop = nn.Dropout(config.embd_pdrop)
        # transformer
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.block_size = config.block_size
        self.apply(self._init_weights)
        self.config = config
        self.no_pos_emb = no_pos_emb
        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, idx, embeddings=None, targets=None):
        # forward the GPT model
        token_embeddings = self.tok_emb(idx) # each index maps to a (learnable) vector

        if embeddings is not None: # prepend explicit embeddings
            token_embeddings = torch.cat((embeddings, token_embeddings), dim=1)

        t = token_embeddings.shape[1]
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."
        position_embeddings = self.pos_emb[:, :t, :] # each position maps to a (learnable) vector
        if self.no_pos_emb==True:
            position_embeddings *=0.
        x = self.drop(token_embeddings + position_embeddings) # dropout
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)
        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

class CondTupleGPT(nn.Module):
    """  the full GPT language model, with a context size of block_size """
    def __init__(self, vocab_sizes, extra_vocab_sizes, block_size, tuple_n, n_layers=(12,), n_head=8, n_embd=256,
                 embd_pdrop=0., resid_pdrop=0., attn_pdrop=0., n_unmasked=0, no_pos_emb=False, cond_emb_same=False, pos_no_restart=False, head_hidden_layers=0):
        """Tuple GPT model

        Args:
            vocab_sizes (tuple(int,*)): how many vocabularies for each element in tuple, e.g. (8192, 4097)
            block_size (int): The maximum sequence length
            n_layers (tuple(int,*), optional): Number of layers. Defaults to 12. e.g. (30, 2)
            n_head (int, optional): head num. Defaults to 8.
            n_embd (int, optional): embedding dimension. Defaults to 256.
            embd_pdrop (float, optional): ?. Defaults to 0..
            resid_pdrop (float, optional): ?. Defaults to 0..
            attn_pdrop (float, optional): ?. Defaults to 0..
            n_unmasked (int, optional): ?. Defaults to 0.
        """
        super().__init__()
        self.__dict__.update(locals())
        self.tok_embs, self.extra_tok_embs = nn.ModuleList([]), nn.ModuleList([])
        self.drops, self.blocks, self.heads = nn.ModuleList([]), nn.ModuleList([]), nn.ModuleList([])
        assert self.tuple_n == len(vocab_sizes)
        assert self.tuple_n == len(n_layers)
        self.extra_tuple_n = len(extra_vocab_sizes)
        for i in range(self.tuple_n):
            vocab_size = vocab_sizes[i]
            n_layer = n_layers[i]
            config = GPTConfig( vocab_size=vocab_size, block_size=block_size,
                                embd_pdrop=embd_pdrop, resid_pdrop=resid_pdrop, attn_pdrop=attn_pdrop,
                                n_layer=n_layer, n_head=n_head, n_embd=n_embd,
                                n_unmasked=n_unmasked, no_pos_emb=False)
            # input embedding stem
            self.tok_embs.append( nn.Embedding(vocab_size, config.n_embd) )
            self.drops.append( nn.Dropout(config.embd_pdrop) )
            # transformer
            self.blocks.append(nn.Sequential(*[Block(config) for _ in range(n_layer)]))
            # decoder heads
            self.heads.append( nn.Sequential(
                                nn.LayerNorm(config.n_embd),
                                *[layer for j in range(head_hidden_layers) 
                                    for layer in 
                                    ( nn.Linear(config.n_embd, config.n_embd), 
                                      nn.ReLU() )
                                ],
                                nn.Linear(config.n_embd, vocab_size, bias=False) 
                               )
                            )
        for i in range(self.extra_tuple_n):
            vocab_size = extra_vocab_sizes[i]
            self.extra_tok_embs.append( nn.Embedding(vocab_size, n_embd) )
        self.pos_emb        = nn.Parameter(torch.zeros(1, block_size, n_embd))
        if cond_emb_same == True:
            self.cond_pos_emb = self.pos_emb
        else:
            self.cond_pos_emb   = nn.Parameter(torch.zeros(1, block_size, n_embd))
        self.block_size = block_size
        self.apply(self._init_weights)
        self.no_pos_emb = no_pos_emb

        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

    def get_block_size(self):
        return self.block_size
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    def get_pos_embeddings(self, L_gen, L_cond):
        if hasattr(self, "pos_no_restart") and self.pos_no_restart==True:
            pos_embd = self.pos_emb[:,       :L_gen+L_cond, :]
        else:
            shape_position_embeddings = self.pos_emb[:,       :L_gen, :] # each position maps to a (learnable) vector
            cond_position_embeddings  = self.cond_pos_emb[:,  :L_cond,  :] # each position maps to a (learnable) vector
            pos_embd = torch.cat([cond_position_embeddings, shape_position_embeddings], axis=1)
        if self.no_pos_emb==True:
            pos_embd *=0.
        return pos_embd
    def get_token_embeddings(self, tok_embs, idx):
        token_embeddings = 0
        for i, tok_emb in enumerate(tok_embs):
            token_embeddings = token_embeddings + tok_emb(idx[..., i])
        return token_embeddings
    def get_embeddings(self, idx, extra_idx, L_cond):
        """ idx: (B, L, tuple_n) cond_idx: idx[:,:L_cond,:] 
            extra_idx: (B, L, extra_tuple_n) 
        """
        assert idx.shape[-1] == self.tuple_n, f"{idx.shape[-1]}!={self.tuple_n}"
        assert extra_idx.shape[-1] == self.extra_tuple_n, f"{extra_idx.shape[-1]}!={self.extra_tuple_n}"
        L = idx.shape[1]
        L_gen = L - L_cond
        assert L <= self.block_size, "Cannot forward, model block size is exhausted."
        # each index maps to a (learnable) vector
        token_embeddings = self.get_token_embeddings( self.tok_embs, idx)
        extra_token_embeddings = self.get_token_embeddings( self.extra_tok_embs, extra_idx)
        pos_embd = self.get_pos_embeddings(L_gen, L_cond)
        # (B, L, embd_dim)
        x = token_embeddings + extra_token_embeddings + pos_embd
        return x
    def compute_logits(self, x, targets):
        """ x: embeddings,  (B, L, embd_dim)
            targets:        (B, L, tuple_n)     """
        logits = [] # list of logits
        for i in range(self.tuple_n):
            x = self.blocks[i]( self.drops[i](x) )
            # (B, L, vocab_size) <- (B, L, embd_dim)
            logits.append( self.heads[i](x) )
            x = x + self.tok_embs[i](targets[...,i]) # targets = idx shifted to left
        return logits
    def sample_next_tuple(self, idx, extra_idx=None, L_cond=1):
        # x: (B, L, embd_dim) <- (B, L_cond, tuple_n), (B, L_gen, tuple_n)
        with torch.no_grad():
            x = self.get_embeddings(idx, extra_idx, L_cond)
            logits = [] # list of logits
            for i in range(self.tuple_n):
                # (B, L, embd_dim)
                x = self.blocks[i]( self.drops[i](x) )
                # (B, L, vocab_size)
                logits.append( self.heads[i](x) )
                # target_i: (B, L), yield: (B, L, vocab_size)
                target_i = yield logits[-1] # sample the logits from outside
                x = x + self.tok_embs[i](target_i) # targets = idx shifted to left
        return logits
    def forward(self, idx, extra_idx=None, L_cond=1, target_idx=None):
        """ idx: (B, L, tuple_n) cond_idx: idx[:,:L_cond,:] 
            extra_idx: (B, L, extra_tuple_n) 
            target_idx: (B, L, tuple_n)
        """

        x = self.get_embeddings(idx, extra_idx, L_cond)
        logits = self.compute_logits(x, target_idx)
        return logits


# unfinished
class SpatialGPT(GPT):
    def __init__(self, cond_vocab_size=1024, cpos_emb_same=True, **kwargs):
        super().__init__(**kwargs)
        self.cond_vocab_size = cond_vocab_size
        self.cpos_emb_same = cpos_emb_same
        tot_size = self.config.vocab_size + cond_vocab_size
        self.tok_emb = nn.Embedding(tot_size, self.config.n_embd)

    def forward(self, idx, cond_idx, targets=None):
        # forward the GPT model
        t = idx.shape[1]
        c = cond_idx.shape[1]
        # 预测可能大于1024，因此会报错
        #assert idx.shape[1]%2==0, f"idx must has length of 2*len(input) instead of length {idx.shape[1]}"
        assert t+c <= self.block_size, "Cannot forward, model block size is exhausted."
        cond_vocab_offsite = self.config.vocab_size
        cond_idx = cond_idx + cond_vocab_offsite # make cond vocab differnet from input vocab
        full_idx = torch.cat((cond_idx, idx), axis=1)
        token_embeddings = self.tok_emb(full_idx) # each index maps to a (learnable) vector
        #if embeddings is not None: # prepend explicit embeddings
        #    token_embeddings = torch.cat((embeddings, token_embeddings), dim=1)
        if self.cpos_emb_same==True:
            position_embeddings = self.pos_emb[:, :t, :] # each position maps to a (learnable) vector
            cond_pos_embd = self.pos_emb[:,:c,:]
            full_pos_emb  = torch.cat((cond_pos_embd, position_embeddings), dim=1)
        else:
            full_pos_emb = self.pos_emb[:,:c+t,:]
       # print(f"t: {position_embeddings.shape}, c: {cond_pos_embd.shape}, full: {full_pos_emb.shape}")
        #print(f"token: {token_embeddings.shape}")
        x = self.drop(token_embeddings + full_pos_emb) # dropout
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)
        #print(f"x shape: {x.shape}, logits shape: {logits.shape}")

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

class DummyGPT(nn.Module):
    # for debugging
    def __init__(self, add_value=1):
        super().__init__()
        self.add_value = add_value

    def forward(self, idx):
        return idx + self.add_value, None


class CodeGPT(nn.Module):
    """Takes in semi-embeddings"""
    def __init__(self, vocab_size, block_size, in_channels, n_layer=12, n_head=8, n_embd=256,
                 embd_pdrop=0., resid_pdrop=0., attn_pdrop=0., n_unmasked=0):
        super().__init__()
        config = GPTConfig(vocab_size=vocab_size, block_size=block_size,
                           embd_pdrop=embd_pdrop, resid_pdrop=resid_pdrop, attn_pdrop=attn_pdrop,
                           n_layer=n_layer, n_head=n_head, n_embd=n_embd,
                           n_unmasked=n_unmasked)
        # input embedding stem
        self.tok_emb = nn.Linear(in_channels, config.n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.drop = nn.Dropout(config.embd_pdrop)
        # transformer
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.block_size = config.block_size
        self.apply(self._init_weights)
        self.config = config
        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, idx, embeddings=None, targets=None):
        # forward the GPT model
        token_embeddings = self.tok_emb(idx) # each index maps to a (learnable) vector

        if embeddings is not None: # prepend explicit embeddings
            token_embeddings = torch.cat((embeddings, token_embeddings), dim=1)

        t = token_embeddings.shape[1]
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."
        position_embeddings = self.pos_emb[:, :t, :] # each position maps to a (learnable) vector
        x = self.drop(token_embeddings + position_embeddings)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss


#### sampling utils

def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float('Inf')
    return out

@torch.no_grad()
def sample(model, x, steps, temperature=1.0, sample=False, top_k=None):
    """
    take a conditioning sequence of indices in x (of shape (b,t)) and predict the next token in
    the sequence, feeding the predictions back into the model each time. Clearly the sampling
    has quadratic complexity unlike an RNN that is only linear, and has a finite context window
    of block_size, unlike an RNN that has an infinite context window.
    """
    block_size = model.get_block_size()
    model.eval()
    for k in range(steps):
        x_cond = x if x.size(1) <= block_size else x[:, -block_size:]  # crop context if needed
        logits, _ = model(x_cond)
        # pluck the logits at the final step and scale by temperature
        logits = logits[:, -1, :] / temperature
        # optionally crop probabilities to only the top k options
        if top_k is not None:
            logits = top_k_logits(logits, top_k)
        # apply softmax to convert to probabilities
        probs = F.softmax(logits, dim=-1)
        # sample from the distribution or take the most likely
        if sample:
            ix = torch.multinomial(probs, num_samples=1)
        else:
            _, ix = torch.topk(probs, k=1, dim=-1)
        # append to the sequence and continue
        x = torch.cat((x, ix), dim=1)

    return x



#### clustering utils

class KMeans(nn.Module):
    def __init__(self, ncluster=512, nc=3, niter=10):
        super().__init__()
        self.ncluster = ncluster
        self.nc = nc
        self.niter = niter
        self.shape = (3,32,32)
        self.register_buffer("C", torch.zeros(self.ncluster,nc))
        self.register_buffer('initialized', torch.tensor(0, dtype=torch.uint8))

    def is_initialized(self):
        return self.initialized.item() == 1

    @torch.no_grad()
    def initialize(self, x):
        N, D = x.shape
        assert D == self.nc, D
        c = x[torch.randperm(N)[:self.ncluster]] # init clusters at random
        for i in range(self.niter):
            # assign all pixels to the closest codebook element
            a = ((x[:, None, :] - c[None, :, :])**2).sum(-1).argmin(1)
            # move each codebook element to be the mean of the pixels that assigned to it
            c = torch.stack([x[a==k].mean(0) for k in range(self.ncluster)])
            # re-assign any poorly positioned codebook elements
            nanix = torch.any(torch.isnan(c), dim=1)
            ndead = nanix.sum().item()
            print('done step %d/%d, re-initialized %d dead clusters' % (i+1, self.niter, ndead))
            c[nanix] = x[torch.randperm(N)[:ndead]] # re-init dead clusters

        self.C.copy_(c)
        self.initialized.fill_(1)


    def forward(self, x, reverse=False, shape=None):
        if not reverse:
            # flatten
            bs,c,h,w = x.shape
            assert c == self.nc
            x = x.reshape(bs,c,h*w,1)
            C = self.C.permute(1,0)
            C = C.reshape(1,c,1,self.ncluster)
            a = ((x-C)**2).sum(1).argmin(-1) # bs, h*w indices
            return a
        else:
            # flatten
            bs, HW = x.shape
            """
            c = self.C.reshape( 1, self.nc,  1, self.ncluster)
            c = c[bs*[0],:,:,:]
            c = c[:,:,HW*[0],:]
            x =      x.reshape(bs,       1, HW,             1)
            x = x[:,3*[0],:,:]
            x = torch.gather(c, dim=3, index=x)
            """
            x = self.C[x]
            x = x.permute(0,2,1)
            shape = shape if shape is not None else self.shape
            x = x.reshape(bs, *shape)

            return x
