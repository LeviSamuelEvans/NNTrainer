import torch
import torch.nn as nn

class Discriminant(nn.Module):
    def __init__(self, d_model):
        super(Discriminant, self).__init__()
        self.fc = nn.Linear(d_model, d_model)

    def forward(self, x, labels):
        logits = self.fc(x)
        probs = torch.softmax(logits, dim=-1)

        labels = labels.long()
        if labels.dim() == 1:
            labels = labels.unsqueeze(-1)

        labels = labels.squeeze(-1)
        labels = labels.unsqueeze(-1).expand(-1, x.size(1)).unsqueeze(-1)

        # gather the probabilities based on the labels
        selected_probs = torch.gather(probs, -1, labels).squeeze(-1)

        return selected_probs

class LearnedPositionalEncodingWithDiscriminant(nn.Module):
    def __init__(self, d_model, dropout, max_len=10):
        super(LearnedPositionalEncodingWithDiscriminant, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pos_embeddings = nn.Embedding(max_len, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.discriminant = Discriminant(d_model)

    def forward(self, x, labels):
        seq_len = x.size(1)
        positions = torch.arange(seq_len, dtype=torch.long, device=x.device).unsqueeze(0).expand(x.size(0), -1)
        position_embeddings = self.pos_embeddings(positions)

        # apply Discriminant
        position_embeddings = self.discriminant(position_embeddings, labels)

        # add extra dimension and repeat the tensor along that dimension
        position_embeddings = position_embeddings.unsqueeze(2).expand(-1, -1, x.size(2))

        x = x + position_embeddings
        x = self.dropout(x)
        x = self.layer_norm(x)
        return x
