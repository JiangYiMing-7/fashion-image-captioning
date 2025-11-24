from __future__ import annotations

from torch import nn

from configs.model_configs import RNNDecoderConfig


class RNNDecoder(nn.Module):
    def __init__(self, config: RNNDecoderConfig) -> None:
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.rnn = nn.LSTM(
            input_size=config.embed_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.fc = nn.Linear(config.hidden_dim, config.vocab_size)

    def forward(self, captions, hidden):
        embeddings = self.dropout(self.embedding(captions))
        outputs, hidden = self.rnn(embeddings, hidden)
        logits = self.fc(outputs)
        return logits, hidden

