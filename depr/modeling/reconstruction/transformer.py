from typing import Any, Optional
from packaging import version
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from ..utils.checkpoint import checkpoint
from ..utils.misc import zero_module, default, exists, Normalize


if version.parse(torch.__version__) >= version.parse("2.0.0"):
    SDP_IS_AVAILABLE = True
    from torch.backends.cuda import SDPBackend, sdp_kernel

    BACKEND_MAP = {
        SDPBackend.MATH: {
            "enable_math": True,
            "enable_flash": False,
            "enable_mem_efficient": False,
        },
        SDPBackend.FLASH_ATTENTION: {
            "enable_math": False,
            "enable_flash": True,
            "enable_mem_efficient": False,
        },
        SDPBackend.EFFICIENT_ATTENTION: {
            "enable_math": False,
            "enable_flash": False,
            "enable_mem_efficient": True,
        },
        None: {"enable_math": True, "enable_flash": True, "enable_mem_efficient": True},
    }
else:
    from contextlib import nullcontext

    SDP_IS_AVAILABLE = False
    sdp_kernel = nullcontext
    BACKEND_MAP = {}
    print(
        f"No SDP backend available, likely because you are running in pytorch versions < 2.0. In fact, "
        f"you are using PyTorch {torch.__version__}. You might want to consider upgrading."
    )

try:
    import xformers
    import xformers.ops

    XFORMERS_IS_AVAILABLE = True
except:
    XFORMERS_IS_AVAILABLE = False
    print("no module 'xformers'. Processing without...")


class CrossAttention(nn.Module):
    def __init__(
        self,
        query_dim,
        context_dim=None,
        heads=8,
        dim_head=64,
        dropout=0.0,
        backend=None,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head**-0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim), nn.Dropout(dropout)
        )
        self.backend = backend

    def forward(
        self,
        x,
        context=None,
        mask=None,
        additional_tokens=None,
        n_times_crossframe_attn_in_self=0,
    ):
        h = self.heads

        if additional_tokens is not None:
            # get the number of masked tokens at the beginning of the output sequence
            n_tokens_to_mask = additional_tokens.shape[1]
            # add additional token
            x = torch.cat([additional_tokens, x], dim=1)

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        if n_times_crossframe_attn_in_self:
            # reprogramming cross-frame attention as in https://arxiv.org/abs/2303.13439
            assert x.shape[0] % n_times_crossframe_attn_in_self == 0
            n_cp = x.shape[0] // n_times_crossframe_attn_in_self
            k = repeat(
                k[::n_times_crossframe_attn_in_self], "b ... -> (b n) ...", n=n_cp
            )
            v = repeat(
                v[::n_times_crossframe_attn_in_self], "b ... -> (b n) ...", n=n_cp
            )

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))

        ## old
        """
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        del q, k

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        sim = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', sim, v)
        """
        ## new
        with sdp_kernel(**BACKEND_MAP[self.backend]):
            # print("dispatching into backend", self.backend, "q/k/v shape: ", q.shape, k.shape, v.shape)
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask
            )  # scale is dim_head ** -0.5 per default

        del q, k, v
        out = rearrange(out, "b h n d -> b n (h d)", h=h)

        if additional_tokens is not None:
            # remove additional token
            out = out[:, n_tokens_to_mask:]
        return self.to_out(out)


class MemoryEfficientCrossAttention(nn.Module):
    # https://github.com/MatthieuTPHR/diffusers/blob/d80b531ff8060ec1ea982b65a1b8df70f73aa67c/src/diffusers/models/attention.py#L223
    def __init__(
        self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0, **kwargs
    ):
        super().__init__()
        print(
            f"Setting up {self.__class__.__name__}. Query dim is {query_dim}, context_dim is {context_dim} and using "
            f"{heads} heads with a dimension of {dim_head}."
        )
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.heads = heads
        self.dim_head = dim_head

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim), nn.Dropout(dropout)
        )
        self.attention_op: Optional[Any] = None

    def forward(
        self,
        x,
        context=None,
        mask=None,
        additional_tokens=None,
        n_times_crossframe_attn_in_self=0,
    ):
        if additional_tokens is not None:
            # get the number of masked tokens at the beginning of the output sequence
            n_tokens_to_mask = additional_tokens.shape[1]
            # add additional token
            x = torch.cat([additional_tokens, x], dim=1)
        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        if n_times_crossframe_attn_in_self:
            # reprogramming cross-frame attention as in https://arxiv.org/abs/2303.13439
            assert x.shape[0] % n_times_crossframe_attn_in_self == 0
            # n_cp = x.shape[0]//n_times_crossframe_attn_in_self
            k = repeat(
                k[::n_times_crossframe_attn_in_self],
                "b ... -> (b n) ...",
                n=n_times_crossframe_attn_in_self,
            )
            v = repeat(
                v[::n_times_crossframe_attn_in_self],
                "b ... -> (b n) ...",
                n=n_times_crossframe_attn_in_self,
            )

        b, _, _ = q.shape
        q, k, v = map(
            lambda t: t.unsqueeze(3)
            .reshape(b, t.shape[1], self.heads, self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b * self.heads, t.shape[1], self.dim_head)
            .contiguous(),
            (q, k, v),
        )
        # actually compute the attention, what we cannot get enough of
        out = xformers.ops.memory_efficient_attention(
            q, k, v, attn_bias=None, op=self.attention_op
        )

        # TODO: Use this directly in the attention operation, as a bias
        if exists(mask):
            raise NotImplementedError
        out = (
            out.unsqueeze(0)
            .reshape(b, self.heads, out.shape[1], self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b, out.shape[1], self.heads * self.dim_head)
        )
        if additional_tokens is not None:
            # remove additional token
            out = out[:, n_tokens_to_mask:]
        return self.to_out(out)


class Lrpe(nn.Module):
    def __init__(
        self,
        num_heads=8,
        embed_dim=64,
    ):
        super().__init__()
        d = num_heads * embed_dim

        self.index = torch.empty(0)
        self.theta = nn.Parameter(
            10000 ** (-2 / d * torch.arange(d)).reshape(num_heads, 1, -1)
        )

    def forward(self, x, offset=0):
        # x: b, h, n, d
        # offset: for k, v cache
        n = x.shape[-2]
        if self.index.shape[0] < n:
            self.index = torch.arange(n).reshape(1, -1, 1).to(x)
        index = self.index[:, :n] + offset
        theta = self.theta * index
        x = torch.concat([x * torch.cos(theta), x * torch.sin(theta)], dim=-1)

        return x


class NormLinearAttention(nn.Module):
    def __init__(
        self,
        query_dim,
        heads,
        dropout=0.0,
        context_dim=0,
        bias=False,
        use_lrpe=True,
        layer=0,
        **kwargs,
    ):
        super().__init__()

        hidden_dim = query_dim
        bias = (bias,)
        self.n_head = heads
        self.use_lrpe = use_lrpe

        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.qkvu_proj = nn.Linear(hidden_dim, 4 * hidden_dim, bias=bias)
        if self.use_lrpe:
            self.lrpe = Lrpe(num_heads=self.n_head, embed_dim=hidden_dim // self.n_head)
        self.act = F.silu
        self.norm = nn.LayerNorm(hidden_dim)
        # self.norm = SimpleRMSNorm(hidden_dim)

        if layer >= 16:
            self.forward_type = "right"
        else:
            self.forward_type = "left"

        self.clip = True
        self.eps = 1e-5

    def abs_clamp(self, t):
        min_mag = 1e-2
        max_mag = 100
        sign = t.sign()
        return t.abs_().clamp_(min_mag, max_mag) * sign

    def forward_right(
        self,
        x,
    ):
        # x: b n d
        n = x.shape[-2]
        # linear map
        q, k, v, u = self.qkvu_proj(x).chunk(4, dim=-1)
        # reshape
        q, k, v = map(
            lambda x: rearrange(x, "b n (h d) -> b h n d", h=self.n_head), [q, k, v]
        )
        # act
        q = self.act(q)
        k = self.act(k)

        # q = F.normalize(q, dim=-1)
        # k = F.normalize(k, dim=-1)

        # lrpe
        if self.use_lrpe:
            offset = 0
            q = self.lrpe(q, offset=offset)
            k = self.lrpe(k, offset=offset)

        kv = torch.einsum("... n d, ... n e -> ... d e", k, v)
        # if self.clip:
        # kv = self.abs_clamp(kv)
        output = torch.einsum("... n d, ... d e -> ... n e", q, kv)

        # Z = 1/(torch.einsum("nlhi,nlhi->nlh", q, k.cumsum(1)) + self.eps)

        # print('right:', q.shape, k.shape, v.shape, kv.shape)

        # reshape
        output = rearrange(output, "b h n d -> b n (h d)")
        # normalize
        output = self.norm(output)

        # gate
        output = u * output
        # outproj
        output = self.out_proj(output)

        return output

    def forward_left(
        self,
        x,
    ):
        # x: b n d
        n = x.shape[-2]
        # linear map
        q, k, v, u = self.qkvu_proj(x).chunk(4, dim=-1)
        # reshape
        q, k, v = map(
            lambda x: rearrange(x, "b n (h d) -> b h n d", h=self.n_head), [q, k, v]
        )
        # act
        q = self.act(q)
        k = self.act(k)

        # q = F.normalize(q, dim=-1)
        # k = F.normalize(k, dim=-1)

        # lrpe
        if self.use_lrpe:
            offset = 0
            q = self.lrpe(q, offset=offset)
            k = self.lrpe(k, offset=offset)

        qk = torch.einsum("... m d, ... n d -> ... m n", q, k)
        # if self.clip:
        # qk = self.abs_clamp(qk)

        output = torch.einsum("... m n, ... n e -> ... m e", qk, v)

        # print('left:', q.shape, k.shape, v.shape, qk.shape)

        # reshape
        output = rearrange(output, "b h n d -> b n (h d)")
        # normalize
        output = self.norm(output)

        # gate
        output = u * output
        # outproj
        output = self.out_proj(output)

        return output

    def forward(
        self, x, context, additional_tokens=None, n_times_crossframe_attn_in_self=0
    ):
        if self.forward_type == "left":
            return self.forward_left(x)
        elif self.forward_type == "right":
            return self.forward_right(x)


class BasicTransformerBlock(nn.Module):
    ATTENTION_MODES = {
        "softmax": CrossAttention,  # vanilla attention
        "softmax-xformers": MemoryEfficientCrossAttention,  # ampere
        "linear": NormLinearAttention,
    }

    def __init__(
        self,
        dim,
        n_heads,
        d_head,
        dropout=0.0,
        context_dim=None,
        gated_ff=True,
        checkpoint=True,
        disable_self_attn=False,
        attn_mode="softmax",
        sdp_backend=None,
        layer=0,
    ):
        super().__init__()
        assert attn_mode in self.ATTENTION_MODES
        if attn_mode != "softmax" and not XFORMERS_IS_AVAILABLE:
            pass
            # print(
            #     f"Attention mode '{attn_mode}' is not available. Falling back to native attention. "
            #     f"This is not a problem in Pytorch >= 2.0. FYI, you are running with PyTorch version {torch.__version__}"
            # )
            # attn_mode = "softmax"
        elif attn_mode == "softmax" and not SDP_IS_AVAILABLE:
            print(
                "We do not support vanilla attention anymore, as it is too expensive. Sorry."
            )
            if not XFORMERS_IS_AVAILABLE:
                assert (
                    False
                ), "Please install xformers via e.g. 'pip install xformers==0.0.16'"
            else:
                print("Falling back to xformers efficient attention.")
                attn_mode = "softmax-xformers"
        attn_cls = self.ATTENTION_MODES[attn_mode]
        if version.parse(torch.__version__) >= version.parse("2.0.0"):
            assert sdp_backend is None or isinstance(sdp_backend, SDPBackend)
        else:
            assert sdp_backend is None
        self.disable_self_attn = disable_self_attn
        self.attn1 = attn_cls(
            query_dim=dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
            context_dim=context_dim if self.disable_self_attn else None,
            backend=sdp_backend,
            layer=layer,
        )  # is a self-attention if not self.disable_self_attn
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = attn_cls(
            query_dim=dim,
            context_dim=context_dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
            backend=sdp_backend,
            layer=layer,
        )  # is self-attn if context is none
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint
        if self.checkpoint:
            print(f"{self.__class__.__name__} is using checkpointing")

    def forward(
        self, x, context=None, additional_tokens=None, n_times_crossframe_attn_in_self=0
    ):
        kwargs = {"x": x}

        if context is not None:
            kwargs.update({"context": context})

        if additional_tokens is not None:
            kwargs.update({"additional_tokens": additional_tokens})

        if n_times_crossframe_attn_in_self:
            kwargs.update(
                {"n_times_crossframe_attn_in_self": n_times_crossframe_attn_in_self}
            )

        # return mixed_checkpoint(self._forward, kwargs, self.parameters(), self.checkpoint)
        if context is None:
            return checkpoint(self._forward, [x], self.parameters(), self.checkpoint)
        else:
            return checkpoint(
                self._forward, [x, context], self.parameters(), self.checkpoint
            )

    def _forward(
        self, x, context=None, additional_tokens=None, n_times_crossframe_attn_in_self=0
    ):
        x = (
            self.attn1(
                self.norm1(x),
                context=context if self.disable_self_attn else None,
                additional_tokens=additional_tokens,
                n_times_crossframe_attn_in_self=(
                    n_times_crossframe_attn_in_self if not self.disable_self_attn else 0
                ),
            )
            + x
        )
        x = (
            self.attn2(
                self.norm2(x), context=context, additional_tokens=additional_tokens
            )
            + x
        )
        x = self.ff(self.norm3(x)) + x
        return x


class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    NEW: use_linear for more efficiency instead of the 1x1 convs
    """

    def __init__(
        self,
        in_channels,
        n_heads,
        d_head,
        depth=1,
        dropout=0.0,
        context_dim=None,
        disable_self_attn=False,
        use_linear=False,
        attn_type="softmax",
        use_checkpoint=True,
        # sdp_backend=SDPBackend.FLASH_ATTENTION
        sdp_backend=None,
        layer=0,
    ):
        super().__init__()
        print(
            f"constructing {self.__class__.__name__} of depth {depth} w/ {in_channels} channels and {n_heads} heads"
        )
        from omegaconf import ListConfig

        if exists(context_dim) and not isinstance(context_dim, (list, ListConfig)):
            context_dim = [context_dim]
        if exists(context_dim) and isinstance(context_dim, list):
            if depth != len(context_dim):
                print(
                    f"WARNING: {self.__class__.__name__}: Found context dims {context_dim} of depth {len(context_dim)}, "
                    f"which does not match the specified 'depth' of {depth}. Setting context_dim to {depth * [context_dim[0]]} now."
                )
                # depth does not match context dims.
                assert all(
                    map(lambda x: x == context_dim[0], context_dim)
                ), "need homogenous context_dim to match depth automatically"
                context_dim = depth * [context_dim[0]]
        elif context_dim is None:
            context_dim = [None] * depth
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)
        if not use_linear:
            self.proj_in = nn.Conv2d(
                in_channels, inner_dim, kernel_size=1, stride=1, padding=0
            )
        else:
            self.proj_in = nn.Linear(in_channels, inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    n_heads,
                    d_head,
                    dropout=dropout,
                    context_dim=context_dim[d],
                    disable_self_attn=disable_self_attn,
                    attn_mode=attn_type,
                    checkpoint=use_checkpoint,
                    sdp_backend=sdp_backend,
                    layer=layer,
                )
                for d in range(depth)
            ]
        )
        if not use_linear:
            self.proj_out = zero_module(
                nn.Conv2d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0)
            )
        else:
            # self.proj_out = zero_module(nn.Linear(in_channels, inner_dim))
            self.proj_out = zero_module(nn.Linear(inner_dim, in_channels))
        self.use_linear = use_linear

    def forward(self, x, context=None):
        # note: if no context is given, cross-attention defaults to self-attention
        if not isinstance(context, list):
            context = [context]
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        if not self.use_linear:
            x = self.proj_in(x)
        x = rearrange(x, "b c h w -> b (h w) c").contiguous()
        if self.use_linear:
            x = self.proj_in(x)
        for i, block in enumerate(self.transformer_blocks):
            if i > 0 and len(context) == 1:
                i = 0  # use same context for each block
            x = block(x, context=context[i])
        if self.use_linear:
            x = self.proj_out(x)
        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w).contiguous()
        if not self.use_linear:
            x = self.proj_out(x)
        return x + x_in


# feedforward
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.0):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = (
            nn.Sequential(nn.Linear(dim, inner_dim), nn.GELU())
            if not glu
            else GEGLU(dim, inner_dim)
        )

        self.net = nn.Sequential(
            project_in, nn.Dropout(dropout), nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)
