import torch

class Residual(torch.nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(dim, dim, 3, 1, 1),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(dim, dim, 3, 1, 1))
        self.alpha = torch.nn.Parameter(torch.rand(dim, 1, 1) * 0.001)

    def forward(self, input):
        return self.layers(input) * self.alpha + input

class SelfAttention(torch.nn.Module):

    def __init__(self, in_dim, num_head, embed_dim):
        super().__init__()
        self.qkv = torch.nn.Sequential(
            torch.nn.Linear(in_dim, 3 * num_head * embed_dim),
            torch.nn.Unflatten(-1, (3, num_head * embed_dim)))
        self.attention = torch.nn.MultiheadAttention(embed_dim=num_head * embed_dim, num_heads=num_head, batch_first=True)
        self.project = torch.nn.Linear(num_head * embed_dim, in_dim)

    def forward(self, input):
        q, k, v = torch.unbind(self.qkv(input), 2)
        output, _ = self.attention(q, k, v)
        return self.project(output) + input

class CrossAttention(torch.nn.Module):

    def __init__(self, in_dim, out_dim, q_dim, num_head, embed_dim):
        super().__init__()
        self.kv = torch.nn.Sequential(
            torch.nn.Linear(in_dim, 2 * num_head * embed_dim),
            torch.nn.Unflatten(-1, (2, num_head * embed_dim)))
        self.q = torch.nn.Linear(q_dim, num_head * embed_dim)
        self.attention = torch.nn.MultiheadAttention(embed_dim=num_head * embed_dim, num_heads=num_head, batch_first=True)
        self.project = torch.nn.Linear(num_head * embed_dim, out_dim)

    def forward(self, input, query):
        q = torch.tile(self.q(query), (input.shape[0], 1, 1))
        k, v = torch.unbind(self.kv(input), 2)
        output, _ = self.attention(q, k, v)
        return self.project(output)

class Model(torch.nn.Module):

    def __init__(self):
        super().__init__()
        def block(in_dim, out_dim):
            return [
                torch.nn.Conv2d(in_dim, out_dim, 1),
                Residual(out_dim),
                torch.nn.MaxPool2d(2)]
        self.layers = torch.nn.Sequential(
            *block(3, 32),
            *block(32, 64),
            torch.nn.Flatten(2))
        self.embedding = torch.nn.Parameter(torch.randn(1, 16 * 16, 64))
        self.encoder = SelfAttention(64, 8, 32)
        self.decoder = CrossAttention(64, 1, 64, 8, 32)
        self.queries = torch.nn.Parameter(torch.randn(1, 1000, 64) * 0.01)

    def forward(self, input):
        buffer = self.layers(input).mT + self.embedding
        buffer = self.encoder(buffer)
        buffer = self.decoder(buffer, self.queries).flatten(1)
        buffer = torch.unflatten(buffer, 1, (200, 5))
        buffer = torch.amax(buffer, -1)
        return buffer

if __name__ == "__main__":
    model = Model()
    image = torch.randn(4, 3, 64, 64)
    output = model(image)
    print(output.shape)
    