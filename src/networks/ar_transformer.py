import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

from typing import Optional

from .layers import FeedForward


class AutoregressiveTransformer(nn.Module):

    def __init__(
        self,
        dim_in: int,
        hidden_channels: int,
        num_blocks: int,
        num_heads: int,
        mlp_ratio: float,
        checkpoint_grads: bool,  # redundant
        drop_attn: float,
        drop_proj: float,
        drop_mlp: float,
        max_len: int,
        use_pos_encoding: bool,
        use_kv_cache: bool,
        generator: nn.Module,
        dim_cond: Optional[int] = None,
        bayesian: bool = False,
    ):

        super().__init__()
        self.dim_in = dim_in
        self.hidden_channels = hidden_channels
        self.max_len = max_len
        self.use_pos_encoding = use_pos_encoding
        self.use_kv_cache = use_kv_cache
        self.generator = generator
        self.bayesian = bayesian

        # input embedding
        self.proj_in = nn.Linear(dim_in, hidden_channels)

        # init start token if no condition will be used
        self.conditional = dim_cond is not None
        if not self.conditional:
            self.start_token = nn.Parameter(torch.randn(1, 1, hidden_channels))

        # init condition embedding if needed to bridge dimensions
        self.use_proj_cond = (dim_cond != hidden_channels) and self.conditional
        if self.use_proj_cond:
            self.proj_cond = nn.Linear(dim_cond, hidden_channels)

        # transformer blocks
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_channels,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    checkpoint_grads=checkpoint_grads,
                    drop_mlp=drop_mlp,
                    drop_attn=drop_attn,
                    drop_proj=drop_proj,
                )
                for _ in range(num_blocks)
            ]
        )
        self.out_norm = nn.LayerNorm(hidden_channels)

        if self.use_pos_encoding:
            # unstructured learnable positional encodings
            self.pos_encoding = nn.Parameter(torch.randn(max_len - 1, hidden_channels))

        # TODO: give learnable encoding to condition token

    def forward(
        self, x: torch.Tensor, pos: int = 0, key_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:

        # forward pass through transformer stack
        for block in self.blocks:
            x = block(x, pos, key_mask=key_mask)  # pos for kv cache at inference
        # x[~mask] = 0
        x = self.out_norm(x)

        return x

    def start(
        self, c: Optional[torch.Tensor] = None, batch_size: int = 1
    ) -> torch.Tensor:
        if c is None:
            assert not self.conditional
            return self.start_token.expand(batch_size, -1, -1)
        c = c.unsqueeze(1)
        return self.proj_cond(c) if self.use_proj_cond else c

    def log_prob(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        key_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        B = x.size(0)

        # embed input subsequences
        x_prev = self.proj_in(x[:, :-1])
        if self.use_pos_encoding:
            # TODO: Return to some kind of encoding? Will have to deal with changing acceptance mask reshape input
            x_prev = x_prev + self.pos_encoding

        x_prev = torch.cat([self.start(c, batch_size=B), x_prev], dim=1)
        cond = self.forward(x_prev, key_mask=key_mask)

        # get logprobs
        log_prob = self.generator.log_prob(
            x.flatten(0, 1).unsqueeze(-1), cond.flatten(0, 1)
        )

        # unflatten sequence dim
        log_prob = log_prob.reshape(B, -1)

        # remove padding
        if key_mask is not None:
            pad_mask = ~key_mask
            log_prob[pad_mask] = 0.0

        return log_prob.mean(1)

    @torch.no_grad()
    def sample_batch(
        self,
        c: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        batch_size: int = 1,
        pad_value: float = 0,
        idx: int = None,
    ) -> torch.Tensor:

        sample = []
        x_prev = self.start(c, batch_size=batch_size)

        batch_size = len(x_prev)
        if self.use_kv_cache:
            self.init_kv_cache(batch_size=batch_size)
        pad_mask = torch.zeros(batch_size, dtype=torch.bool, device=x_prev.device)

        for i in range(self.max_len):

            # sample next element, given last transformer output
            cond = self.forward(x_prev, pos=i)
            x_next = self.generator.sample_batch(c=cond[:, -1], idx=i)

            # pad stopped events
            if lengths is not None:
                stop_mask = i == lengths
                pad_mask = pad_mask | stop_mask

                if pad_mask.all():  # break if every event is finished
                    break

                x_next[pad_mask] = pad_value

            # store sample
            sample.append(x_next.squeeze(1))

            if i < self.max_len - 1:
                # embed latest sample and append to input
                x_embed = self.proj_in(x_next)
                if self.use_pos_encoding:
                    x_embed = x_embed + self.pos_encoding[i]
                if self.use_kv_cache:
                    x_prev = x_embed.unsqueeze(1)
                else:
                    x_prev = torch.cat([x_prev, x_embed.unsqueeze(1)], dim=1)

        sample = torch.stack(sample, dim=1)

        pad_len = self.max_len - sample.size(1)
        sample = F.pad(sample, (0, 0, 0, pad_len), value=pad_value)

        self.clean_kv_cache()

        return sample

    def init_kv_cache(self, batch_size):
        # install KV cache in all the Attention layers
        for block in self.blocks:
            layer_dtype = block.attn.qkv.weight.dtype
            layer_device = block.attn.qkv.weight.device
            num_heads = block.attn.num_heads
            block.attn.cache = KVCache(
                batch_size=batch_size,
                seq_length=self.max_len,
                n_kv_heads=num_heads,
                head_dim=self.hidden_channels // num_heads,
                dtype=layer_dtype,
                device=layer_device,
            )

    def clean_kv_cache(self):
        # clean up the KV cache in all the layers
        for block in self.blocks:
            block.attn.cache = None


class StringBreakMixtureModel(nn.Module):

    def __init__(self, size, dim_cond, num_gaussians):
        self.size = size
        super().__init__()
        self.proj_cat1 = nn.Linear(dim_cond, 2)  # fromPos
        self.proj_cat2 = nn.Linear(dim_cond, 2)  # mass
        self.proj_gaussian = nn.Linear(
            dim_cond, 3 * num_gaussians
        )  # continuous features
        self.num_gaussians = num_gaussians

    def _extract_gmm_params(self, cond, min_sigmaarg=-4, max_sigmaarg=5):
        logits = self.proj_gaussian(cond)
        logits = logits.view(*logits.shape[:-1], self.num_gaussians, 3)

        weights = F.softmax(logits[..., 2], dim=-1)
        mu = logits[..., 0]

        # avoid inf and 0 (unstable in D.Normal)
        sigmaarg = torch.clamp(logits[..., 1], min=min_sigmaarg, max=max_sigmaarg)
        sigma = torch.exp(sigmaarg)

        return mu, sigma, weights

    def build_gmm(self, cond):
        mu, sigma, weights = self._extract_gmm_params(cond)
        mix = D.Categorical(probs=weights, validate_args=False)
        normal = D.Normal(mu, sigma, validate_args=False)
        return D.MixtureSameFamily(mix, normal, validate_args=False)

    def build_cat(self, logits):
        return D.Categorical(probs=F.softmax(logits, dim=-1), validate_args=False)

    def log_prob(self, x, cond):

        x = x.view(-1, self.size)
        cond = cond.view(-1, self.size, cond.size(-1))

        cat1 = self.build_cat(self.proj_cat1(cond[:, [0]]))
        cat2 = self.build_cat(self.proj_cat2(cond[:, [1]]))
        gmm = self.build_gmm(cond[:, 2:])
        log_prob = torch.hstack(
            [cat1.log_prob(x[:, [0]]), cat2.log_prob(x[:, [1]]), gmm.log_prob(x[:, 2:])]
        ).view(-1)

        return log_prob

    def sample_batch(self, c: torch.Tensor, idx: int) -> torch.Tensor:
        if idx == 0:
            cat1 = self.build_cat(self.proj_cat1(c))
            sample = cat1.sample(()).float()
        elif idx == 1:
            cat2 = self.build_cat(self.proj_cat2(c))
            sample = cat2.sample(()).float()
        else:
            gmm = self.build_gmm(c)
            sample = gmm.sample(())

        return sample.unsqueeze(1)


class GaussianMixtureModel(nn.Module):

    def __init__(self, dim_cond, num_gaussians):
        super().__init__()
        self.num_gaussians = num_gaussians
        self.proj = nn.Linear(dim_cond, 3 * num_gaussians)

    def _extract_params(self, logits, min_sigmaarg=-3.5, max_sigmaarg=5):
        logits = logits.view(*logits.shape[:-1], self.num_gaussians, 3)

        weights = F.softmax(logits[..., 2], dim=-1)
        mu = logits[..., 0]

        # avoid inf and 0 (unstable in D.Normal)
        sigmaarg = torch.clamp(logits[..., 1], min=min_sigmaarg, max=max_sigmaarg)
        sigma = torch.exp(sigmaarg)

        return mu, sigma, weights

    def build_gmm(self, logits):
        mu, sigma, weights = self._extract_params(logits)
        mix = D.Categorical(probs=weights, validate_args=False)
        normal = D.Normal(mu, sigma, validate_args=False)
        gmm = D.MixtureSameFamily(mix, normal, validate_args=False)
        return gmm

    def log_prob(self, x, cond):
        logits = self.proj(cond)
        x = x.squeeze()
        gmm = self.build_gmm(logits)
        return gmm.log_prob(x)

    def sample_batch(self, c, **kwargs):
        logits = self.proj(c)
        gmm = self.build_gmm(logits)
        return gmm.sample(()).unsqueeze(1)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        drop_mlp: float = 0.0,
        is_causal: bool = False,
        checkpoint_grads: bool = False,
        **attn_kwargs,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            is_causal=is_causal,
            **attn_kwargs,
        )
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.ffwd = FeedForward(dim=hidden_size, mult=mlp_ratio, dropout=drop_mlp)
        self.checkpoint_grads = checkpoint_grads

    def forward(
        self, x: torch.Tensor, pos: int = 0, key_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), pos, key_mask=key_mask)
        x = x + self.ffwd(self.norm2(x))
        return x


class Attention(nn.Module):

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        is_causal: bool = False,
        drop_attn: float = 0.0,
        drop_proj: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.is_causal = is_causal
        self.drop_attn = nn.Dropout(drop_attn)
        self.proj = nn.Linear(dim, dim)
        self.drop_proj = nn.Dropout(drop_proj)

        # placeholder for KVCache object to be utilized at inference
        self.cache = None

    def forward(
        self, x: torch.Tensor, pos: int = 0, key_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:

        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)  # (batch_size, num_heads, seq_len, head_dim)

        if self.cache is not None:
            # update the KV cache with current KV and get all the previous KVs
            k, v = self.cache.update(pos, k, v)

        q, k = self.q_norm(q), self.k_norm(k)

        attn_mask = (
            None
            if key_mask is None
            else (key_mask[..., None] & key_mask[..., None, :]).tril().unsqueeze(1)
        )
        
        x = F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=(attn_mask is None) and (self.cache is None),
            attn_mask=attn_mask,
            dropout_p=self.drop_attn.p if self.training else 0.0,
        )

        # if len(q)==1000:
        #     print(f"{q[0,0,:,0]=}")

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.drop_proj(x)
        return x


class KVCache(nn.Module):
    def __init__(self, batch_size, seq_length, n_kv_heads, head_dim, dtype, device):
        super().__init__()
        cache_shape = (batch_size, n_kv_heads, seq_length, head_dim)
        self.register_buffer(
            "cache_k", torch.zeros(cache_shape, dtype=dtype, device=device)
        )
        self.register_buffer(
            "cache_v", torch.zeros(cache_shape, dtype=dtype, device=device)
        )

    def update(self, pos, k, v):
        seqlen = k.size(2)
        self.cache_k[:, :, pos : pos + seqlen] = k
        self.cache_v[:, :, pos : pos + seqlen] = v
        k = self.cache_k[:, :, : pos + seqlen]
        v = self.cache_v[:, :, : pos + seqlen]
        return k, v
