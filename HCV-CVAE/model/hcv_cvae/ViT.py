import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from typing import Optional, Tuple


# ============================================================
# 1) Patch Embedding（增强：ConvStem + InputProj + PosEmbedding）
# ============================================================
class PatchEmbedding(nn.Module):
    def __init__(self,
                 image_width=620,
                 image_height=460,
                 patch_size=20,
                 in_channels=3,
                 embed_dim=512,
                 dropout=0.,
                 use_conv_stem=True):
        super().__init__()
        assert image_width % patch_size == 0 and image_height % patch_size == 0
        self.grid_w = image_width // patch_size
        self.grid_h = image_height // patch_size
        n_patches = self.grid_w * self.grid_h
        self.use_conv_stem = use_conv_stem

        # ---- ConvStem ----
        if use_conv_stem:
            mid = max(embed_dim // 4, 32)
            self.conv_stem = nn.Sequential(
                nn.Conv2d(in_channels, mid, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid, embed_dim, 3, 1, 1)
            )
            # 🔧 新增 InputProj：统一通道数以便残差相加
            self.input_proj = nn.Conv2d(in_channels, embed_dim, 1)
            stem_in = embed_dim
        else:
            stem_in = in_channels

        # ---- Patch Embedding ----
        self.patch_embedding = nn.Conv2d(stem_in, embed_dim,
                                         kernel_size=patch_size,
                                         stride=patch_size)

        # ---- Tokens & Pos Embedding ----
        self.position_embeddings = nn.Parameter(torch.randn(1, n_patches + 2, embed_dim))
        self.mu_tokens = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.log_var_tokens = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        B = x.size(0)
        if self.use_conv_stem:
            x = self.conv_stem(x) + self.input_proj(x)

        x = self.patch_embedding(x)                 # [B, D, H', W']
        x = x.flatten(2).transpose(1, 2)            # [B, N, D]

        mu_tok = self.mu_tokens.expand(B, -1, -1)
        lv_tok = self.log_var_tokens.expand(B, -1, -1)
        x = torch.cat([mu_tok, lv_tok, x], dim=1)

        x = x + self.position_embeddings
        x = self.dropout(x)
        return x, self.grid_h, self.grid_w


# ============================================================
# 2) 相对位置偏置（兼容旧版 PyTorch）
# ============================================================
class RelativePositionBias2D(nn.Module):
    def __init__(self, num_heads: int, grid_h: int, grid_w: int):
        super().__init__()
        self.num_heads = num_heads
        self.grid_h = grid_h
        self.grid_w = grid_w
        num_relative = (2 * grid_h - 1) * (2 * grid_w - 1)
        self.relative_bias_table = nn.Parameter(torch.zeros(num_heads, num_relative))

        # 坐标索引映射
        coords_h = torch.arange(grid_h)
        coords_w = torch.arange(grid_w)
        try:
            coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))
        except TypeError:  # 兼容 torch < 1.10
            coords = torch.stack(torch.meshgrid(coords_h, coords_w))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += grid_h - 1
        relative_coords[:, :, 1] += grid_w - 1
        relative_coords[:, :, 0] *= (2 * grid_w - 1)
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        nn.init.trunc_normal_(self.relative_bias_table, std=0.02)

    def forward(self):
        bias = self.relative_bias_table[:, self.relative_position_index.view(-1)]
        bias = bias.view(self.num_heads,
                         self.grid_h * self.grid_w,
                         self.grid_h * self.grid_w)
        return bias


# ============================================================
# 3) Attention（带相对位置偏置）
# ============================================================
class Attention(nn.Module):
    def __init__(self,
                 embed_dim,
                 num_heads,
                 attn_head_size=None,
                 qkv_bias=True,
                 dropout=0.,
                 attention_dropout=0.,
                 use_rel_pos_bias=True,
                 grid_hw=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.attn_head_size = embed_dim // num_heads if attn_head_size is None else attn_head_size
        self.scale = self.attn_head_size ** -0.5
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)
        self.out = nn.Linear(embed_dim, embed_dim)
        self.attn_dropout = nn.Dropout(attention_dropout)
        self.proj_dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)

        self.rel_pos = None
        if use_rel_pos_bias and grid_hw is not None:
            gh, gw = grid_hw
            self.rel_pos = RelativePositionBias2D(num_heads, gh, gw)

    def _multihead(self, x):
        B, N, _ = x.size()
        x = x.view(B, N, self.num_heads, self.attn_head_size).permute(0, 2, 1, 3)
        return x

    def forward(self, x):
        B, N, _ = x.shape
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = map(self._multihead, qkv)
        attn = (q @ k.transpose(-2, -1)) * self.scale

        if self.rel_pos is not None and N > 2:
            rel_bias = self.rel_pos()
            attn[:, :, 2:, 2:] += rel_bias.unsqueeze(0)

        attn = self.softmax(attn)
        attn = self.attn_dropout(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        out = self.out(out)
        out = self.proj_dropout(out)
        return out


# ============================================================
# 4) MLP + SEBlock
# ============================================================
class Mlp(nn.Module):
    def __init__(self, embed_dim, mlp_ratio=2., dropout=0.):
        super().__init__()
        hidden = int(embed_dim * mlp_ratio)
        self.fc1 = nn.Linear(embed_dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, embed_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.fc2(self.act(self.fc1(x))))


class SEBlock(nn.Module):
    def __init__(self, dim, reduction=8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim, dim // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(dim // reduction, dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        w = self.fc(x.mean(1))[:, None, :]
        return x * w


# ============================================================
# 5) Encoder Layer
# ============================================================
class EncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, attn_head_size=None, qkv_bias=True,
                 mlp_ratio=2.0, dropout=0., attention_dropout=0.,
                 use_rel_pos_bias=True, grid_hw=None, use_se=True):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = Attention(embed_dim, num_heads, attn_head_size, qkv_bias,
                              dropout, attention_dropout,
                              use_rel_pos_bias, grid_hw)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = Mlp(embed_dim, mlp_ratio, dropout)
        self.use_se = use_se
        if use_se:
            self.se = SEBlock(embed_dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        if self.use_se:
            x = self.se(x)
        return x


# ============================================================
# 6) Encoder
# ============================================================
class Encoder(nn.Module):
    def __init__(self, embed_dim, num_heads, depth, attn_head_size=None,
                 qkv_bias=True, mlp_ratio=2.0, dropout=0., attention_dropout=0.,
                 use_rel_pos_bias=True, grid_hw=None, use_se=True):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(embed_dim, num_heads, attn_head_size, qkv_bias,
                         mlp_ratio, dropout, attention_dropout,
                         use_rel_pos_bias, grid_hw, use_se)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x, return_intermediate=True):
        feats = []
        for blk in self.layers:
            x = blk(x)
            if return_intermediate:
                feats.append(x)
        x = self.norm(x)
        return (x, feats) if return_intermediate else x


# ============================================================
# 7) VisualTransformer（增强版）
# ============================================================
class VisualTransformer(nn.Module):
    def __init__(self,
                 image_width=256,
                 image_height=256,
                 patch_size=16,
                 in_channels=3,
                 embed_dim=512,
                 depth=4,
                 num_heads=8,
                 attn_head_size=None,
                 mlp_ratio=2.0,
                 qkv_bias=True,
                 dropout=0.,
                 attention_dropout=0.,
                 num_dim=100,
                 use_conv_stem=True,
                 use_rel_pos_bias=True,
                 use_se=True,
                 use_hierarchical_latent=True):
        super().__init__()

        self.patch_embedding = PatchEmbedding(
            image_width, image_height, patch_size,
            in_channels, embed_dim, dropout,
            use_conv_stem
        )

        grid_h = image_height // patch_size
        grid_w = image_width // patch_size
        self.encoder = Encoder(embed_dim, num_heads, depth,
                               attn_head_size, qkv_bias, mlp_ratio,
                               dropout, attention_dropout,
                               use_rel_pos_bias, (grid_h, grid_w), use_se)

        self.use_hierarchical_latent = use_hierarchical_latent
        self.mu = nn.Linear(embed_dim, num_dim)
        self.log_var = nn.Linear(embed_dim, num_dim)

        if use_hierarchical_latent:
            self.mu_mid = nn.Linear(embed_dim, num_dim)
            self.log_var_mid = nn.Linear(embed_dim, num_dim)
            self.mu_global = nn.Linear(embed_dim, num_dim)
            self.log_var_global = nn.Linear(embed_dim, num_dim)

    def forward(self, x):
        tokens, gh, gw = self.patch_embedding(x)
        out, feats = self.encoder(tokens, return_intermediate=True)

        mu_base = self.mu(out[:, 0])
        lv_base = self.log_var(out[:, 1])

        if not self.use_hierarchical_latent or len(feats) < 2:
            return mu_base, lv_base

        mid_feat = feats[len(feats)//2]
        mid_mu = self.mu_mid(mid_feat[:, 0])
        mid_lv = self.log_var_mid(mid_feat[:, 1])

        patch_feat = out[:, 2:, :].mean(1)
        g_mu = self.mu_global(patch_feat)
        g_lv = self.log_var_global(patch_feat)

        mu = (mu_base + mid_mu + g_mu) / 3
        log_var = (lv_base + mid_lv + g_lv) / 3
        return mu, log_var
