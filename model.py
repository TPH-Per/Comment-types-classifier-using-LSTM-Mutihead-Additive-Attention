import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        d = hidden_dim * 2
        self.head_projs  = nn.ModuleList([nn.Linear(d, d//num_heads) for _ in range(num_heads)])
        self.head_scores = nn.ModuleList([nn.Linear(d//num_heads, 1) for _ in range(num_heads)])
        self.output_proj = nn.Linear(d * num_heads, d)
        self.dropout     = nn.Dropout(dropout)
        self.layer_norm  = nn.LayerNorm(d)

    def forward(self, lstm_out, mask):
        head_ctxs, head_wts = [], []
        for proj, scorer in zip(self.head_projs, self.head_scores):
            h      = torch.tanh(proj(lstm_out))
            scores = scorer(h).squeeze(-1)
            scores = scores.masked_fill(~mask, -1e9)
            wts    = self.dropout(F.softmax(scores, dim=1))
            ctx    = torch.bmm(wts.unsqueeze(1), lstm_out).squeeze(1)
            head_ctxs.append(ctx)
            head_wts.append(wts)
        multi = torch.cat(head_ctxs, dim=-1)
        out   = self.layer_norm(self.output_proj(multi))
        avg_w = torch.stack(head_wts).mean(0)
        return out, avg_w


class BiLSTMSentiment(nn.Module):
    def __init__(self, vocab_size, emb_dim=300, hidden_dim=256,
                 num_layers=2, num_classes=3, num_heads=4,
                 dropout=0.5, pretrained_emb=None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        if pretrained_emb is not None:
            self.embedding.weight = nn.Parameter(pretrained_emb)
            print('  OK: FastText weights loaded')

        self.lstm = nn.LSTM(
            emb_dim, hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )
        self.attention  = MultiHeadAttention(hidden_dim, num_heads, dropout=0.1)
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.6),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, ids, mask):
        emb         = self.dropout(self.embedding(ids))
        lstm_out, _ = self.lstm(emb)
        lstm_out    = self.dropout(lstm_out)
        ctx, attn   = self.attention(lstm_out, mask)
        return self.classifier(ctx), attn
