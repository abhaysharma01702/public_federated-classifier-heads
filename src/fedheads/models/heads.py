"""Classification heads.

Every head takes the encoder's token states (batch, tokens, hidden) and the attention mask
(1 for real tokens, 0 for padding) and returns (logits, pooled representation). Padding never
changes a head's output: means, maxima, attention and the LSTM all skip it (tested).

| Name        | Design                                                                      |
|-------------|-----------------------------------------------------------------------------|
| linear      | masked mean -> dropout -> linear (the baseline)                              |
| linear_cls  | <s> token -> dropout -> linear                     |
| cnn         | the old FedBLACX CNN branch: conv(k 3, 128 filters) -> ReLU -> masked max     |
| bilstm      | the old FedBLACX BiLSTM branch: BiLSTM(256 each way) -> masked mean           |
| cnn_bilstm  | the old FedBLACX head: both branches, concatenated (128 + 512)               |
| mlp         | masked mean -> MLP whose width matches the cnn_bilstm parameter count         |
| attention   | additive attention (128 units) over tokens -> weighted sum -> linear         |

All heads share one dropout value (model.dropout), so they differ only in design.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

CNN_FILTERS = 128      # old FedBLACX head (notebooks/Model.ipynb)
CNN_KERNEL = 3
LSTM_UNITS = 256       # per direction, old FedBLACX head
ATTENTION_UNITS = 128


def masked_mean(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean over non-padding positions. Padding never changes the result."""
    m = mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)


def masked_max(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Maximum over non-padding positions, per feature."""
    return x.masked_fill(mask.unsqueeze(-1) == 0, torch.finfo(x.dtype).min).max(dim=1).values


class _Classifier(nn.Module):
    """Dropout, then a linear layer to the classes."""

    def __init__(self, in_features: int, n_classes: int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(in_features, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.dropout(x))


class LinearHead(nn.Module):
    """Masked mean pooling, dropout, then a linear layer to the 3 classes."""

    def __init__(self, hidden_size: int, n_classes: int = 3, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size, n_classes)

    def forward(self, hidden: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pooled = masked_mean(hidden, mask)
        return self.linear(self.dropout(pooled)), pooled


class LinearClsHead(LinearHead):
    """The <s> token (position 0), dropout, then a linear layer."""

    def forward(self, hidden: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pooled = hidden[:, 0]
        return self.linear(self.dropout(pooled)), pooled


class _CNNBranch(nn.Module):
    """Conv1d over token states, ReLU, masked max over tokens. Padding states are zeroed first,
    so the last real token's window sees zeros, exactly as at the end of an unpadded sequence."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.conv = nn.Conv1d(hidden_size, CNN_FILTERS, CNN_KERNEL, padding=CNN_KERNEL // 2)

    def forward(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = (hidden * mask.unsqueeze(-1).to(hidden.dtype)).transpose(1, 2)
        c = torch.relu(self.conv(x)).transpose(1, 2)
        return masked_max(c, mask)


class _BiLSTMBranch(nn.Module):
    """Bidirectional LSTM on the real tokens only (packed), then masked mean over tokens."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.lstm = nn.LSTM(hidden_size, LSTM_UNITS, batch_first=True, bidirectional=True)

    def forward(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        lengths = mask.sum(dim=1).clamp(min=1).cpu()
        packed = pack_padded_sequence(hidden, lengths, batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=hidden.shape[1])
        return masked_mean(out, mask)


class CNNHead(nn.Module):
    def __init__(self, hidden_size: int, n_classes: int = 3, dropout: float = 0.1):
        super().__init__()
        self.cnn = _CNNBranch(hidden_size)
        self.classifier = _Classifier(CNN_FILTERS, n_classes, dropout)

    def forward(self, hidden, mask):
        pooled = self.cnn(hidden, mask)
        return self.classifier(pooled), pooled


class BiLSTMHead(nn.Module):
    def __init__(self, hidden_size: int, n_classes: int = 3, dropout: float = 0.1):
        super().__init__()
        self.bilstm = _BiLSTMBranch(hidden_size)
        self.classifier = _Classifier(2 * LSTM_UNITS, n_classes, dropout)

    def forward(self, hidden, mask):
        pooled = self.bilstm(hidden, mask)
        return self.classifier(pooled), pooled


class CNNBiLSTMHead(nn.Module):
    """The old FedBLACX head: CNN and BiLSTM branches side by side, concatenated."""

    def __init__(self, hidden_size: int, n_classes: int = 3, dropout: float = 0.1):
        super().__init__()
        self.cnn = _CNNBranch(hidden_size)
        self.bilstm = _BiLSTMBranch(hidden_size)
        self.classifier = _Classifier(CNN_FILTERS + 2 * LSTM_UNITS, n_classes, dropout)

    def forward(self, hidden, mask):
        # float() on both: under fp16 autocast the branches can return different dtypes.
        pooled = torch.cat([self.cnn(hidden, mask).float(), self.bilstm(hidden, mask).float()], dim=1)
        return self.classifier(pooled), pooled


def matched_mlp_width(hidden_size: int, n_classes: int = 3) -> int:
    """Width w that brings the MLP (hidden*w + w + w*n_classes + n_classes parameters) closest
    to the cnn_bilstm head's parameter count, for this hidden size."""
    target = sum(p.numel() for p in CNNBiLSTMHead(hidden_size, n_classes).parameters())
    per_unit = hidden_size + 1 + n_classes
    w = (target - n_classes) / per_unit
    lo, hi = int(w), int(w) + 1
    return lo if abs(lo * per_unit + n_classes - target) <= abs(hi * per_unit + n_classes - target) else hi


class MLPHead(nn.Module):
    """Masked mean, dropout, linear to `width`, GELU, dropout, linear to the classes. The width
    defaults to the one that matches the cnn_bilstm head's parameter count."""

    def __init__(self, hidden_size: int, n_classes: int = 3, dropout: float = 0.1, width: int | None = None):
        super().__init__()
        self.width = width or matched_mlp_width(hidden_size, n_classes)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, self.width)
        self.classifier = _Classifier(self.width, n_classes, dropout)

    def forward(self, hidden, mask):
        pooled = masked_mean(hidden, mask)
        return self.classifier(torch.nn.functional.gelu(self.fc1(self.dropout(pooled)))), pooled


class AttentionHead(nn.Module):
    """Additive attention over tokens (score = v . tanh(W h)), softmax over real tokens only,
    weighted sum of token states, then dropout and a linear layer."""

    def __init__(self, hidden_size: int, n_classes: int = 3, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(hidden_size, ATTENTION_UNITS)
        self.score = nn.Linear(ATTENTION_UNITS, 1, bias=False)
        self.classifier = _Classifier(hidden_size, n_classes, dropout)

    def weights(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        s = self.score(torch.tanh(self.proj(hidden))).squeeze(-1).float()
        return torch.softmax(s.masked_fill(mask == 0, float("-inf")), dim=1)

    def forward(self, hidden, mask):
        w = self.weights(hidden, mask).to(hidden.dtype)
        pooled = (w.unsqueeze(-1) * hidden).sum(dim=1)
        return self.classifier(pooled), pooled


HEADS = {
    "linear": LinearHead,
    "linear_cls": LinearClsHead,
    "cnn": CNNHead,
    "bilstm": BiLSTMHead,
    "cnn_bilstm": CNNBiLSTMHead,
    "mlp": MLPHead,
    "attention": AttentionHead,
}
MAIN_HEADS = ["linear", "cnn", "bilstm", "cnn_bilstm", "mlp", "attention"]   # linear_cls is a side check


def build_head(name: str, hidden_size: int, n_classes: int = 3, dropout: float = 0.1, **kwargs) -> nn.Module:
    if name not in HEADS:
        raise ValueError(f"unknown head {name!r}; known: {sorted(HEADS)}")
    return HEADS[name](hidden_size, n_classes, dropout, **kwargs)
