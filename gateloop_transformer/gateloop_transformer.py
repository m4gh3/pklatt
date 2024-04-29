from functools import partial

import torch
from torch.nn import Module, ModuleList
from torch import nn, einsum, Tensor
from torch.utils.checkpoint import checkpoint
import torch.nn.functional as F

from einops import rearrange
from einops.layers.torch import Rearrange

#from rotary_embedding_torch import RotaryEmbedding

from gateloop_transformer.associative_scan import associative_scan

import math

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def Sequential(*modules):
    modules = list(filter(exists, modules))
    num_modules = len(modules)

    if num_modules == 0:
        return nn.Identity()
    elif num_modules == 1:
        return modules[0]

    return nn.Sequential(*modules)

def quad_feat_map(x):
    #return torch.cat([0.75*(x.unsqueeze(-1)*x.unsqueeze(-2)).flatten(start_dim=-2), 1.3*x ], dim=-1 )
    #return torch.cat([0.72*(x.unsqueeze(-1)*x.unsqueeze(-2)).flatten(start_dim=-2), 1.06*x, torch.ones(x.shape[:-1]+(1,)).to(x.device)], dim=-1 )
    alpha = torch.linalg.vector_norm(x, dim=-1 ).unsqueeze(-1)
    x = F.normalize(x, dim=-1 )
    return torch.cat([(1-0.5**alpha)*x, torch.ones(x.shape[:-1]+(1,)).to(x.device)], dim=-1 )


# rms norm

class RMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.normalize(x, dim = -1) * self.scale * self.gamma

# norm wrappers

class PreNorm(Module):
    def __init__(self, dim, fn: Module):
        super().__init__()
        self.fn = fn
        self.norm = RMSNorm(dim)

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs) + x

class PostNorm(Module):
    def __init__(self, dim, fn: Module):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, **kwargs):
        return self.norm(self.fn(x, **kwargs) + x)

class CaConv1d(Module):
    def __init__(self, in_ch, out_ch, ker_sz, groups=1, bias=True ):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, ker_sz, padding=ker_sz-1 )

    def forward(self, x ):
        s_len = x.size(1)
        return (self.conv(x.permute(0,2,1))[...,:s_len]).permute(0,2,1)

# feedforward

def FeedForward(dim, mult = 4):
    dim_inner = dim * mult
    return nn.Sequential(
        nn.Linear(dim, dim_inner),
        nn.GELU(),
        nn.Linear(dim_inner, dim)
    )

# data gated linear attention with "gateloop operator"

def pklatt_op(q, k, v ):

    #F.normalize(q, dim=-1 )
    #F.normalize(k, dim=-1 )
    q = quad_feat_map(q)
    k = quad_feat_map(k)

    kv = einsum('b n d, b n e -> b n d e', k, v ).cumsum(1)
    k_ = k.cumsum(1)
    qk_ = einsum('bnd,bnd->bn', q, k_ ).unsqueeze(-1)
    qkv = einsum('bnd,bnde->bne', q, kv )

    return (qkv)/(qk_)



class GateLoopedAttention(Module):
    def __init__(
        self,
        dim,
        heads = None,
        dim_inner = None,
        checkpoint_gate_looped_attn = True,
        add_swish_gating = True,
        sub_ln = False,
        frac_gradient_state_transition = 0.9
    ):
        super().__init__()
        self.frac_gradient_state_transition = frac_gradient_state_transition
        self.checkpoint_gate_looped_attn = checkpoint_gate_looped_attn

        dim_inner = default(dim_inner, dim)
        heads = default(heads, dim_inner)

        self.heads = heads
        assert (dim_inner % heads) == 0, f'dimension for gate looped attention {dim_inner} must be divisible by number of gate loop heads {heads}'

        self.split_heads = Rearrange('b n (h d) -> (b h) n d', h = heads)

        #self.rotary_emb = RotaryEmbedding(dim_inner//heads) 

        self.to_qkv = CaConv1d(dim, dim_inner * 3, 5, bias=False ) #nn.Linear(dim, dim_inner * 3, bias = False)

        #self.to_a = nn.Sequential(
        #    nn.Linear(dim, heads * 2),
        #    Rearrange('b n (h c) -> (b h) n 1 1 c', h = heads, c = 2)
        #)

        self.merge_heads = Rearrange('(b h) n d -> b n (h d)', h = heads)

        self.maybe_sub_ln = nn.LayerNorm(dim_inner) if sub_ln else nn.Identity()

        self.to_gates = None

        if add_swish_gating:
            self.to_gates = nn.Sequential(
                nn.Linear(dim, dim_inner, bias = False),
                nn.SiLU()
            )

        self.to_out = nn.Linear(dim_inner, dim, bias = False) if dim_inner != dim or add_swish_gating else nn.Identity()
        self.hlstm = nn.LSTM(dim_inner//heads, dim_inner//heads, dim_inner//heads, batch_first=True )

    def forward(
        self,
        x,
        ablate_complex = False,
        ablate_state_transition = False
    ):
        #print(x.shape)
        frac_gradient = self.frac_gradient_state_transition

        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
  
        q, k, v = map(self.split_heads, (q, k, v))

        #q = self.rotary_emb.rotate_queries_or_keys(q)
        #k = self.rotary_emb.rotate_queries_or_keys(k)

        need_backwards = any([t.requires_grad for t in (q, k, v )])

        fn = partial(checkpoint, pklatt_op ) if need_backwards and self.checkpoint_gate_looped_attn else pklatt_op

        out = fn(q, k, v )
        #print(out.shape)
        out_, _ = self.hlstm(out)
        out = out_ + out

        out = self.merge_heads(out)

        out = self.maybe_sub_ln(out)

        if exists(self.to_gates):
            out = self.to_gates(x) * out
        #out_, _ = self.to_out0(out)
        #out = out + out_
        out = self.to_out(out)

        return out

# main class

class Transformer(Module):
    def __init__(
        self,
        dim,
        *,
        num_tokens,
        depth,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        checkpoint_gate_looped_attn = True,
        use_gate_looped_attn = True,
        gate_loop_heads = None,
        attn_add_swish_gating = True,
        dim_gate_looped_attn = None,
        attn_softmax_normalize = None,
        data_dependent_rel_pos = False,
        frac_gradient_state_transition = 0.9,
        ablate_complex = False,
        ablate_state_transition = False,
        rotary_emb = False,
        post_ln_norm = False,
        sub_ln = False
    ):
        super().__init__()
        self.ablate_complex = ablate_complex
        self.ablate_state_transition = ablate_state_transition

        self.token_emb = nn.Embedding(num_tokens, dim)

        layers = ModuleList([])

        layer_wrapper = PreNorm if not post_ln_norm else PostNorm

        for _ in range(depth):

            spatial_mixer = GateLoopedAttention(
                dim = dim,
                heads = gate_loop_heads,
                dim_inner = dim_gate_looped_attn,
                add_swish_gating = attn_add_swish_gating,
                sub_ln = sub_ln,
                checkpoint_gate_looped_attn = checkpoint_gate_looped_attn,
                frac_gradient_state_transition = frac_gradient_state_transition
            ) 
            
            channelwise_mixer = FeedForward(
                dim = dim,
                mult = ff_mult
            )

            layers.append(ModuleList([
                layer_wrapper(dim, spatial_mixer),
                layer_wrapper(dim, channelwise_mixer)
            ]))

        self.layers = ModuleList(layers)

        self.to_logits = Sequential(
            RMSNorm(dim) if not post_ln_norm else None,
            nn.Linear(dim, num_tokens, bias = False)
        )

    def forward(
        self,
        x,
        return_loss = False,
        ablate_complex = None,
        ablate_state_transition = None
    ):
        ablate_complex = default(ablate_complex, self.ablate_complex)
        ablate_state_transition = default(ablate_state_transition, self.ablate_state_transition)

        if return_loss:
            x, labels = x[:, :-1], x[:, 1:]

        x = self.token_emb(x)

        for attn, ff in self.layers:
            x = attn(
                x,
                ablate_complex = ablate_complex,
                ablate_state_transition = ablate_state_transition
            )

            x = ff(x)

        logits = self.to_logits(x)

        if not return_loss:
            return logits

        logits = rearrange(logits, 'b n c -> b c n')
        return F.cross_entropy(logits, labels)
