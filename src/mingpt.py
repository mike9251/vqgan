import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.embed_dim % config.n_head == 0
        self.key = nn.Linear(config.embed_dim, config.embed_dim)
        self.query = nn.Linear(config.embed_dim, config.embed_dim)
        self.value = nn.Linear(config.embed_dim, config.embed_dim)

        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)

        self.proj = nn.Linear(config.embed_dim, config.embed_dim)

        mask = torch.tril(torch.ones(config.block_size, config.block_size))

        if config.n_unmasked > 0:
            mask[: config.n_unmasked, : config.n_unmasked] = 1

        self.register_buffer("mask", mask.view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head

    def forward(self, x, layer_past=None):
        B, T, C = x.shape

        k = (
            self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        )  # B, n_head, T, head_embed
        q = (
            self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        )  # B, n_head, T, head_embed
        v = (
            self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        )  # B, n_head, T, head_embed

        present = torch.stack((k, v))
        if layer_past is not None:
            prev_key, prev_value = layer_past
            # append to the sequence
            k = torch.cat((prev_key, k), dim=-2)
            v = torch.cat((prev_value, v), dim=-2)

        att = (q @ k.transpose(-2, -1)) * (k.shape[-1] ** -0.5)
        if layer_past is None:
            att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))

        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v  # B, n_heads, T, head_size
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        y = self.resid_drop(self.proj(y))
        return y, present


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.embed_dim)
        self.ln2 = nn.LayerNorm(config.embed_dim)
        self.attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.embed_dim, 4 * config.embed_dim),
            nn.GELU(),
            nn.Linear(config.embed_dim * 4, config.embed_dim),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x, layer_past=None, return_present=False):
        if return_present:
            assert not self.training

        attn, present = self.attn(self.ln1(x), layer_past=layer_past)

        x = x + attn

        x = x + self.mlp(self.ln2(x))
        if layer_past is not None or return_present:
            return x, present

        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.tok_embed = nn.Embedding(config.vocab_size, config.embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, config.block_size, config.embed_dim))
        self.drop = nn.Dropout(config.embed_pdrop)

        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layers)])

        self.ln = nn.LayerNorm(config.embed_dim)
        self.head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)

        self.config = config

        self.apply(self._init_weights)

    def get_block_size(self):
        return self.config.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, idx, embedings=None):
        token_embed = self.tok_embed(idx)

        if embedings is not None:
            token_embed = torch.cat((embedings, token_embed), dim=1)

        t = token_embed.shape[1]

        assert t <= self.config.block_size

        position_embed = self.pos_embed[:, :t, :]

        x = self.drop(token_embed + position_embed)
        x = self.blocks(x)
        x = self.ln(x)
        logits = self.head(x)

        return logits
