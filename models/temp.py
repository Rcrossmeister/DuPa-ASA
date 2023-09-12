import torch
import torch.nn as nn

class AdditiveAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(AdditiveAttention, self).__init__()
        self.W1 = nn.Linear(input_dim, hidden_dim)
        self.W2 = nn.Linear(input_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1)

    def forward(self, x, q=None):
        # x: [batch_size, seq_len, input_dim]
        # q: [batch_size, input_dim], default: None
        # W1: [input_dim, hidden_dim]
        # W2: [input_dim, hidden_dim]
        # v: [hidden_dim, 1]

        if q is None:
            q = torch.zeros(x.shape[0], x.shape[2]).to(x.device)

        # Compute the similarity scores between q and each element in x
        scores = self.v(torch.tanh(self.W1(x) + self.W2(q).unsqueeze(1)))

        # Normalize the scores to get a probability distribution
        alpha = torch.softmax(scores, dim=1)

        # Compute the weighted sum of the input elements using the attention scores
        context = torch.sum(alpha * x, dim=1)

        return context, alpha
