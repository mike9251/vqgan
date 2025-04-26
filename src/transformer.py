import torch
import torch.nn as nn
import torch.nn.functional as F
from mingpt import GPT
from vqgan import VQGAN

import logging
logging.basicConfig(filename=None, encoding='utf-8', level=logging.DEBUG)


class VQGANTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.sos_token = config.sos_token

        self.vqgan = VQGAN(config)
        self._load_vqgan_weights(config.vqgan_weights)
        self.vqgan.eval()

        self.transformer = GPT(config)
        
        self.pkeep = config.pkeep
        self.device = config.device

        if "transformer_weights" in config:
            self._load_transformer_weights(config.transformer_weights)

    def _load_vqgan_weights(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        self.vqgan.load_state_dict(ckpt["vqgan"])
        logging.info(f"VQGAN loaded from {ckpt_path}")
    
    def _load_transformer_weights(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        self.transformer.load_state_dict(ckpt["transformer"])
        logging.info(f"Transformer loaded from {ckpt_path}")
    
    @torch.no_grad()
    def encode_to_z(self, x):
        z_q, min_emb_indices, _ = self.vqgan.encode(x)
        b, c, h, w = z_q.shape # b, emb, h, w
        # min_emb_indices: b, h, w
        min_emb_indices = min_emb_indices.view(b, -1) # b, t

        return min_emb_indices
    
    @torch.no_grad()
    def token_to_image(self, idx, p1=16, p2=16):
        z_q = self.vqgan.codebook.embedding(idx).reshape(idx.shape[0], p1, p2, 256)
        z_q = z_q.permute(0, 3, 1, 2)
        img = self.vqgan.decode(z_q)
        return img

    def forward(self, x):
        idx = self.encode_to_z(x)

        sos_token_idx = torch.ones((x.shape[0], 1), device=x.device) * self.sos_token # B, 1
        sos_token_idx = sos_token_idx.long()

        mask = torch.bernoulli(self.pkeep * torch.ones_like(idx)).long() # sample elements from {0, 1} with pkeep

        # for elements mask == 0 generate random indices
        rand_idx = torch.randint_like(idx, self.transformer.config.vocab_size)
        new_idx = idx * mask + (1 - mask) * rand_idx
        new_idx = torch.cat((sos_token_idx, new_idx), dim=1)
        
        target = idx
        logits = self.transformer(new_idx[:, :-1])

        return logits, target

    def top_k_logits(self, logits, topk):
        v, ix = torch.topk(logits, topk)

        out = logits.clone()

        out[out < v[..., [-1]]] = float("-inf")
        return out
    
    @torch.no_grad()
    def sample(self, x, c, steps, temperature=1.0, top_k=100):
        self.transformer.eval()

        if x is not None:
            x = torch.cat((c, x), dim=1)
        else:
            x = c

        for i in range(steps):
            logits = self.transformer(x)
            logits = logits[:, -1, :] / temperature

            if top_k > 0:
                logits = self.top_k_logits(logits, top_k)
            
            probs = F.softmax(logits, dim=-1)

            ix = torch.multinomial(probs, num_samples=1)

            x = torch.cat((x, ix), dim=-1)
        
        x = x[:, c.shape[1]:]
        
        self.transformer.train()
        
        return x
    
    @torch.no_grad()
    def log_imgs(self, x):
        idx = self.encode_to_z(x)

        sos_token_idx = self.sos_token * torch.ones((idx.shape[0], 1), device=idx.device)
        sos_token_idx = sos_token_idx.long()

        start_idx = idx[:, :idx.shape[1] // 2]
        idx_half_sample = self.sample(start_idx, sos_token_idx, idx.shape[1] - start_idx.shape[1])
        img_half_sample = self.token_to_image(idx_half_sample)
        
        idx_full_sample = self.sample(None, sos_token_idx, idx.shape[1])
        img_full_sample = self.token_to_image(idx_full_sample)

        img_rec = self.token_to_image(idx)

        return x, img_rec, img_half_sample, img_full_sample

    @torch.no_grad()
    def generate(self, steps = 256):
        sos_token_idx = self.sos_token * torch.ones((1, 1), device=self.device).long()
        
        idx_full_sample = self.sample(None, sos_token_idx, steps)

        return self.token_to_image(idx_full_sample, p1=int(steps**0.5), p2=int(steps**0.5))



