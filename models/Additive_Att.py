import torch
import torch.nn as nn
class AdditiveAttention(nn.Module):
    def __init__(self, config):
        super(AdditiveAttention, self).__init__()
        self.config = config
        self.input_dim = config.hidden_size

        self.query = nn.Linear(self.input_dim, self.input_dim)
        self.key = nn.Linear(self.input_dim, self.input_dim)
        self.attention_weights = nn.Linear(self.input_dim, 1)
        self.transform = nn.Linear(self.input_dim, self.input_dim)

        self.softmax = nn.Softmax(dim=-1)

        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, hidden_states, attention_mask):
        # batch_size, seq_len, hidden_size
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)

        # batch_size, seq_len, hidden_size
        attention_inputs = mixed_query_layer.unsqueeze(2) + mixed_key_layer.unsqueeze(1)
        attention_scores = self.attention_weights(torch.tanh(attention_inputs)).squeeze(-1)

        # add attention mask
        attention_scores += attention_mask

        # batch_size, seq_len
        attention_probs = self.softmax(attention_scores)

        # batch_size, seq_len, hidden_size
        weighted_value = torch.matmul(attention_probs.unsqueeze(1), hidden_states).squeeze(1)
        weighted_value = self.transform(weighted_value) + mixed_query_layer

        return weighted_value
