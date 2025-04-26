import torch
import torch.nn as nn
import torch.nn.functional as F


class GroupNorm(nn.Module):
    def __init__(self, num_channels, num_groups=32):
        super().__init__()
        self.gn = nn.GroupNorm(num_groups, num_channels, eps=1e-6, affine=True)

    def forward(self, x):
        return self.gn(x)
    

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.block = nn.Sequential(
            GroupNorm(in_channels),
            Swish(),
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            GroupNorm(out_channels),
            Swish(),
            # add dropout here
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
        )

        if in_channels != out_channels:
            self.channel_up = nn.Conv2d(in_channels, out_channels, 1, 1, 0)

    
    def forward(self, x):
        if self.in_channels != self.out_channels:
            return self.block(x) + self.channel_up(x)

        return x + self.block(x)
    

class UpBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, 1, 1)
    
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


class DownBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # strided conv
        self.conv = nn.Conv2d(channels, channels, 3, 2, 0)
    
    def forward(self, x):
        pad = (0, 1, 0, 1)
        x = F.pad(x, pad, mode="constant", value=0)
        return self.conv(x)


class NonLocalBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.in_channels = channels
        self.gn = GroupNorm(channels)
        # make it one Conv2d
        self.q = nn.Conv2d(channels, channels, 1, 1, 0)
        self.k = nn.Conv2d(channels, channels, 1, 1, 0)
        self.v = nn.Conv2d(channels, channels, 1, 1, 0)

        self.qkv = nn.Conv2d(channels, 3 * channels, 1, 1, 0)
        
        self.proj = nn.Conv2d(channels, channels, 1, 1, 0)
    
    def forward2(self, x):
        h_ = x
        h_ = self.gn(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        q = q.reshape(b,c,h*w)
        q = q.permute(0,2,1)   # b,hw,c
        k = k.reshape(b,c,h*w) # b,c,hw
        w_ = torch.bmm(q,k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = F.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b,c,h*w)
        w_ = w_.permute(0,2,1)   # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v,w_)     # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b,c,h,w)

        h_ = self.proj(h_)

        return x+h_
    

    def forward(self, x):
        y = self.gn(x)
        q, k, v = self.q(y), self.k(y), self.v(y)

        b, c, h, w = q.shape

        q = q.reshape(b, c, h * w) # b, c hw
        q = q.permute(0, 2, 1) # b, hw, c
        k = k.reshape(b, c, h * w)
        v = v.reshape(b, c, h * w)

        att = q @ k # b, hw, hw

        att = att * (c**-0.5)
        att = F.softmax(att, dim=-1)
        att = att.permute(0, 2, 1)
    
        out =  v @ att # b, hw, c

        out = out.reshape(b, c, h, w)

        out = self.proj(out)

        return x + out
    



if __name__ == "__main__":
    gn = GroupNorm(128, 32)
    x = torch.rand((6, 128, 256, 256))
    y = gn(x)
    print(y.shape)

    @torch.no_grad
    def test():
        device = torch.device("mps")

        q = torch.rand((4, 128, 16, 16), device=device)
        k = torch.rand((4, 128, 16, 16), device=device)
        v = torch.rand((4, 128, 16, 16), device=device)

        b, c, h, w = q.shape

        q1 = q.clone().reshape(b, c, h * w) # b, c hw
        q1 = q1.permute(0, 2, 1) # b, hw, c
        k1 = k.clone().reshape(b, c, h * w)
        v1 = v.clone().reshape(b, c, h * w)

        att1 = q1 @ k1 # b, hw, hw

        att1 = att1 * (c**-0.5)
        att1 = F.softmax(att1, dim=-1)
        att1 = att1.permute(0, 2, 1)
        print(f"v1={v1.shape}, att1={att1.shape}")
        # verify this!!!
        # att = att.permute(0, 2, 1) # softmax if distributed along vertical dim
        y1 =  v1 @ att1 # b, hw, c

        q2 = q.clone().reshape(b, c, h*w)
        q2 = q2.permute(0, 2, 1)
        k2 = k.clone().reshape(b, c, h*w)
        v2 = v.clone().reshape(b, c, h*w)

        attn2 = torch.bmm(q2, k2)
        attn2 = attn2 * (int(c)**(-0.5))
        attn2 = F.softmax(attn2, dim=2)
        attn2 = attn2.permute(0, 2, 1)

        A = torch.bmm(v2, attn2)

        print(torch.allclose(att1, attn2))

        print(y1.shape, A.shape)
        print(torch.allclose(y1, A))
    
    test()

    q = nn.Conv2d(2, 3, 1, 1, 0, bias=False)
    k = nn.Conv2d(2, 3, 1, 1, 0, bias=False)
    v = nn.Conv2d(2, 3, 1, 1, 0, bias=False)

    qkv = nn.Conv2d(2, 3 * 3, 1, 1, 0, bias=False)
    qkv.requires_grad_(False)
    w = torch.cat([q.weight.data, k.weight.data, v.weight.data], dim=0)

    qkv.weight.copy_(w)

    print(w.shape)

    print(torch.allclose(w, qkv.weight.data))

    x = torch.rand((4, 2, 16, 16))

    q1, k1, v1 = q(x), k(x), v(x)

    q2, k2, v2 = torch.chunk(qkv(x), 3, dim=1)

    print(torch.allclose(q1, q2))
    print(torch.allclose(k1, k2))
    print(torch.allclose(v1, v2))