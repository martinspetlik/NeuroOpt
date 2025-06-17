import math
import torch
from torch import nn, einsum
import torch.nn.functional as F
from functools import partial
from einops import rearrange
from einops_exts import rearrange_many

from rotary_embedding_torch import RotaryEmbedding


def exists(x):
    return x is not None


def is_odd(n):
    return (n % 2) == 1


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device=device, dtype=torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device=device, dtype=torch.bool)
    else:
        return torch.zeros(shape, device=device).float().uniform_(0, 1) < prob


# Module for adding relative position bias in attention mechanism
class RelativePositionBias(nn.Module):
    def __init__(self, heads=8, num_buckets=32, max_distance=128):
        super().__init__()
        self.num_buckets = num_buckets  # number of buckets to quantize relative distances
        self.max_distance = max_distance  # maximum distance to consider
        self.relative_attention_bias = nn.Embedding(num_buckets, heads)  # embedding for each bucket

    @staticmethod
    def _relative_position_bucket(relative_position, num_buckets=32, max_distance=128):
        # Maps relative positions to buckets, handling small and large distances separately
        ret = 0
        n = -relative_position  # reverse order for attention

        num_buckets //= 2  # split buckets for negative and positive distances
        ret += (n < 0).long() * num_buckets  # use upper half of buckets for negative positions
        n = torch.abs(n)

        max_exact = num_buckets // 2  # exact mapping for small distances
        is_small = n < max_exact

        # Logarithmic mapping for larger distances
        val_if_large = max_exact + (
                torch.log(n.float() / max_exact) / math.log(max_distance / max_exact)
                * (num_buckets - max_exact)
        ).long()
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))

        ret += torch.where(is_small, n, val_if_large)  # use small or large mapping
        return ret

    def forward(self, n, device):
        # Create relative position matrix
        q_pos = torch.arange(n, dtype=torch.long, device=device)
        k_pos = torch.arange(n, dtype=torch.long, device=device)
        rel_pos = rearrange(k_pos, 'j -> 1 j') - rearrange(q_pos, 'i -> i 1')

        # Bucket the relative positions
        rp_bucket = self._relative_position_bucket(
            rel_pos, num_buckets=self.num_buckets, max_distance=self.max_distance
        )

        # Get biases from embedding
        values = self.relative_attention_bias(rp_bucket)
        return rearrange(values, 'i j h -> h i j')  # reshape for attention heads


# Wrapper to add residual connection to any module
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn  # wrapped function

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x  # add residual connection


# Sinusoidal positional embeddings (used in transformers or diffusion models)
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim  # output dimension of embedding

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]  # broadcast to shape (batch, half_dim)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)  # concatenate sin and cos
        return emb


# Upsample via transposed convolution (double spatial size)
def upsample(dim):
    return nn.ConvTranspose2d(dim, dim, kernel_size=4, stride=2, padding=1)


# Downsample via strided convolution (halve spatial size)
def downsample(dim):
    return nn.Conv2d(dim, dim, kernel_size=4, stride=2, padding=1)


# Custom LayerNorm for 2D feature maps (over channels)
class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, dim, 1, 1))  # learnable scaling

    def forward(self, x):
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.gamma


# Applies normalization before passing input to the given function
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)


# Basic building block: Conv -> GroupNorm -> Activation
class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, kernel_size=3, padding=1)  # 3x3 convolution
        self.norm = nn.GroupNorm(groups, dim_out)  # group normalization
        self.act = nn.SiLU()  # SiLU activation (a.k.a. swish)

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        # FiLM-style conditioning (Feature-wise Linear Modulation)
        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        return self.act(x)


# Residual block with optional FiLM-style conditioning using time embeddings
class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()

        # MLP to generate scale and shift from time embedding
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)

        # Optional 1x1 conv if input and output dimensions differ
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        scale_shift = None
        if exists(self.mlp):
            assert exists(time_emb), 'time emb must be passed in'
            time_emb = self.mlp(time_emb)  # (batch, 2 * dim_out)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')  # reshape to broadcast over spatial dims
            scale_shift = time_emb.chunk(2, dim=1)  # split into scale and shift

        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)  # residual connection


# Efficient spatial attention using linearized attention
class SpatialLinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5  # scaling factor for stability
        self.heads = heads
        hidden_dim = dim_head * heads

        # Project input to queries, keys, and values using 1x1 convolutions
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        # Final output projection
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape  # batch, channels, height, width
        # Split QKV from a single convolution
        qkv = self.to_qkv(x).chunk(3, dim=1)

        # Rearrange: split into heads and flatten spatial dims
        q, k, v = rearrange_many(qkv, 'b (h c) x y -> b h c (x y)', h=self.heads)

        # Normalize queries and keys (along different axes)
        q = q.softmax(dim=-2)  # softmax across channels
        k = k.softmax(dim=-1)  # softmax across spatial positions

        # Scale the queries
        q = q * self.scale

        # Compute context: efficient linear attention mechanism
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)
        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)

        # Rearrange back to (B, C, H, W)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h=self.heads, x=h, y=w)
        return self.to_out(out)


# A wrapper that reshapes input/output tensors before and after applying an inner module
class EinopsToAndFrom(nn.Module):
    def __init__(self, from_einops: str, to_einops: str, fn: nn.Module):
        super().__init__()
        self.from_einops = from_einops  # original shape pattern
        self.to_einops = to_einops      # shape pattern for inner function
        self.fn = fn                    # inner function to apply

    def forward(self, x, **kwargs):
        # Create a mapping of shape names to actual sizes
        shape_dict = dict(zip(self.from_einops.split(), x.shape))

        # Rearrange to the expected shape for the function
        x = rearrange(x, f'{self.from_einops} -> {self.to_einops}')
        x = self.fn(x, **kwargs)  # apply inner function
        # Rearrange back to the original shape
        x = rearrange(x, f'{self.to_einops} -> {self.from_einops}', **shape_dict)
        return x


# Standard multi-head self-attention for sequences
class Attention(nn.Module):
    def __init__(
        self,
        dim,                  # input dimension
        heads=4,              # number of attention heads
        dim_head=32,          # dimension per head
        rotary_emb=None       # optional rotary embeddings
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.rotary_emb = rotary_emb

        # Linear layers to project input into queries, keys, and values
        self.to_qkv = nn.Linear(dim, hidden_dim * 3, bias=False)
        # Output projection
        self.to_out = nn.Linear(hidden_dim, dim, bias=False)

    def forward(
        self,
        x,                   # shape: (batch, sequence, dim)
        pos_bias=None,       # optional positional bias
        focus_present_mask=None  # optional masking for attention
    ):
        b, n, _ = x.shape
        device = x.device

        # Project input into Q, K, V
        qkv = self.to_qkv(x).chunk(3, dim=-1)

        # If all tokens are focused (e.g., masked), return values directly
        if exists(focus_present_mask) and focus_present_mask.all():
            values = qkv[-1]
            return self.to_out(values)

        # Rearrange QKV for multi-head attention
        q, k, v = rearrange_many(qkv, 'b n (h d) -> b h n d', h=self.heads)

        # Scale queries
        q = q * self.scale

        # Apply rotary embeddings if present
        if exists(self.rotary_emb):
            q = self.rotary_emb.rotate_queries_or_keys(q)
            k = self.rotary_emb.rotate_queries_or_keys(k)

        # Compute similarity (dot product between queries and keys)
        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        # Add positional bias if provided
        if exists(pos_bias):
            sim = sim + pos_bias

        # If a focus mask is present and not all are masked
        if exists(focus_present_mask) and not (~focus_present_mask).all():
            attend_all_mask = torch.ones((n, n), device=device, dtype=torch.bool)
            attend_self_mask = torch.eye(n, device=device, dtype=torch.bool)

            # Apply selective masking based on focus
            mask = torch.where(
                rearrange(focus_present_mask, 'b -> b 1 1 1'),
                attend_self_mask[None, None, :, :],
                attend_all_mask[None, None, :, :]
            )

            # Mask out disallowed positions
            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)

        # Numerically stable softmax
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        # Weighted sum of values
        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        # Merge heads and project output
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)



def match_spatial_size_2d(x, skip):
    """
    Adjust spatial size of x to match skip (for 2D inputs).
    x, skip: tensors with shape (B, C, H, W)
    Returns x cropped or padded to (H_skip, W_skip).
    """
    skip_h, skip_w = skip.shape[2], skip.shape[3]
    x_h, x_w = x.shape[2], x.shape[3]

    # Crop x if larger than skip
    x = x[:, :, :skip_h, :skip_w]

    # Pad x if smaller than skip
    pad_h = max(skip_h - x.shape[2], 0)
    pad_w = max(skip_w - x.shape[3], 0)
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, pad_w, 0, pad_h))  # pad right, bottom

    return x


class UNetAdvanced(nn.Module):
    def __init__(
        self,
        dim,                            # Base dimensionality of the model
        cond_dim=None,                 # Optional conditioning vector dimension
        out_dim=None,                  # Number of output channels
        dim_mults=(1, 2, 4, 8),        # Multiplicative factors for each level of the UNet
        channels=3,                    # Input channel size (e.g., 3 for RGB)
        attn_heads=8,                  # Number of attention heads
        attn_dim_head=32,              # Dimensionality of each attention head
        init_dim=None,                 # Optional override for the initial convolution dim
        init_kernel_size=7,            # Size of the kernel in the initial conv layer
        use_sparse_linear_attn=True,   # Whether to use sparse linear spatial attention
        resnet_groups=8,               # Number of groups for GroupNorm in ResNet blocks
        use_rotary_emb=True,           # Whether to use rotary positional embeddings
        use_temporal_attention=True):  # Whether to use temporal attention
        super().__init__()

        self.channels = channels
        self.dim = dim
        self._use_temporal_attention = use_temporal_attention

        print("self._use_temporal_attention ", self._use_temporal_attention)
        print("attn heads ", attn_heads)
        print("attn_dim_head", attn_dim_head)

        # Rotary positional embedding for temporal attention
        rotary_emb = RotaryEmbedding(min(32, attn_dim_head)) if use_rotary_emb else None

        # Helper function for creating temporal attention blocks
        def temporal_attn(dim):
            return EinopsToAndFrom('b c h w', 'b (h w) c', Attention(
                dim, heads=attn_heads, dim_head=attn_dim_head, rotary_emb=rotary_emb))

        # Relative position bias for temporal attention (for 32 frames max)
        self.time_rel_pos_bias = RelativePositionBias(
            heads=attn_heads, max_distance=32)

        # Initial convolution layer to map input to feature space
        init_dim = default(init_dim, dim)
        assert is_odd(init_kernel_size)
        init_padding = init_kernel_size // 2
        self.init_conv = nn.Conv2d(channels, init_dim, kernel_size=7, padding=init_padding)

        # Optional initial temporal attention
        self.init_temporal_attn = Residual(PreNorm(init_dim, temporal_attn(init_dim)))

        # Feature map dimensions per stage
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # Time embedding MLP
        time_dim = dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])

        num_resolutions = len(in_out)

        # ResNet block template
        block_klass = partial(ResnetBlock, groups=resnet_groups)
        block_klass_cond = partial(block_klass, time_emb_dim=cond_dim)

        # Downsampling path
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                block_klass_cond(dim_in, dim_out),         # First ResNet block
                block_klass_cond(dim_out, dim_out),        # Second ResNet block
                Residual(PreNorm(dim_out, SpatialLinearAttention(
                    dim_out, heads=attn_heads))) if use_sparse_linear_attn else nn.Identity(),
                Residual(PreNorm(dim_out, temporal_attn(dim_out))),  # Temporal attention
                downsample(dim_out) if not is_last else nn.Identity()  # Downsample unless last
            ]))

        # Middle block
        mid_dim = dims[-1]
        self.mid_block1 = block_klass_cond(mid_dim, mid_dim)

        # Mid-level spatial and temporal attention
        spatial_attn = EinopsToAndFrom(
            'b c h w', 'b (h w) c', Attention(mid_dim, heads=attn_heads))
        self.mid_spatial_attn = Residual(PreNorm(mid_dim, spatial_attn))
        self.mid_temporal_attn = Residual(PreNorm(mid_dim, temporal_attn(mid_dim)))
        self.mid_block2 = block_klass_cond(mid_dim, mid_dim)

        # Upsampling path
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(nn.ModuleList([
                block_klass_cond(dim_out * 2, dim_in),  # Concatenated input, so dim * 2
                block_klass_cond(dim_in, dim_in),
                Residual(PreNorm(dim_in, SpatialLinearAttention(
                    dim_in, heads=attn_heads))) if use_sparse_linear_attn else nn.Identity(),
                Residual(PreNorm(dim_in, temporal_attn(dim_in))),
                upsample(dim_in) if not is_last else nn.Identity()
            ]))

        # Final convolution to desired output channels
        out_dim = default(out_dim, channels)
        self.final_conv = nn.Sequential(
            block_klass(dim * 2, dim),  # Combine with residual skip connection
            nn.Conv2d(dim, out_dim, 1)
        )

    def forward(self, x, time=None, focus_present_mask=None):
        x, cond = x  # `x` is a tuple (input, condition)
        x = x.float()

        # Compute relative position bias
        time_rel_pos_bias = self.time_rel_pos_bias(x.shape[2], device=x.device)

        # Initial convolution and residual clone
        x = self.init_conv(x)
        r = x.clone()

        # Optional temporal attention at the input
        if self._use_temporal_attention:
            x = self.init_temporal_attn(x, pos_bias=time_rel_pos_bias)

        # Time embedding
        t = self.time_mlp(time) if exists(self.time_mlp) else None

        h = []  # Skip connections

        # Downsampling blocks
        for block1, block2, spatial_attn, temporal_attn, downsample in self.downs:
            x = block1(x, t)
            x = block2(x, t)
            x = spatial_attn(x)
            if self._use_temporal_attention:
                x = temporal_attn(x, pos_bias=time_rel_pos_bias,
                                  focus_present_mask=focus_present_mask)
            h.append(x)
            x = downsample(x)

        # Middle blocks
        x = self.mid_block1(x, t)
        x = self.mid_spatial_attn(x)
        if self._use_temporal_attention:
            x = self.mid_temporal_attn(
                x, pos_bias=time_rel_pos_bias, focus_present_mask=focus_present_mask)
        x = self.mid_block2(x, t)

        # Upsampling blocks
        for block1, block2, spatial_attn, temporal_attn, upsample in self.ups:
            skip = h.pop()
            x = match_spatial_size_2d(x, skip)  # Resize x to match skip connection size
            x = torch.cat((x, skip), dim=1)     # Concatenate skip features
            x = block1(x, t)
            x = block2(x, t)
            x = spatial_attn(x)
            if self._use_temporal_attention:
                x = temporal_attn(x, pos_bias=time_rel_pos_bias,
                                  focus_present_mask=focus_present_mask)
            x = upsample(x)

        # Final output projection
        x = torch.cat((x, r), dim=1)
        return self.final_conv(x)

