import torch
import torch.nn as nn


class Codebook(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_vectors = config["num_codebook_vectors"]
        self.latent_dim = config["latent_dim"]
        self.beta = config["beta"]

        self.embedding = nn.Embedding(
            num_embeddings=self.num_vectors, embedding_dim=self.latent_dim
        )  # num_vectors, latent_dim
        self.embedding.weight.data.uniform_(-1.0 / self.num_vectors, 1.0 / self.num_vectors)

    def forward(self, z: torch.Tensor):
        z = z.permute(0, 2, 3, 1)  # num channels == latent_dim
        z_flat = z.reshape(-1, self.latent_dim)  # bhw, latent_dim

        # compute distances between all z_flat and every vector in embedding
        # this is just an expanded formula for (a - b)**2 = a**2 + b**2 - 2ab
        d = (
            torch.sum(z_flat**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1)
            - 2 * (torch.matmul(z_flat, self.embedding.weight.t()))
        )  # bhw, 1

        min_emb_indices = torch.argmin(d, dim=1)

        # sample the codebook
        z_q = self.embedding(min_emb_indices).view(
            z.shape
        )  # bhw, latent_dim -> b, h, w, latent_dim

        # first term pushes the encoder to produce embeddings similar to code book - codebook loss
        # if we make a K-means analogy, this loss term is similar to centroid update step.
        # second term pushes the codebook to be similar to encoder's embedings - commitment loss
        loss = torch.mean((z_q.detach() - z) ** 2) + self.beta * torch.mean(
            (z_q - z.detach()) ** 2
        )

        # Straight-through gradient estimation
        z_q = z + (z_q - z).detach()

        z_q = z_q.permute(0, 3, 1, 2)

        return z_q, min_emb_indices, loss


if __name__ == "__main__":
    device = torch.device("mps")

    codebook = Codebook({"num_codebook_vectors": 100, "latent_dim": 100, "beta": 0.5})
    codebook = codebook.to(device)

    z = torch.rand((4, 100, 16, 16), device=device)

    z_q, ind, loss = codebook(z)

    print(z_q.shape, loss)
