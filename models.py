"""Model definitions for EEG electrode ablation study."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttentionEEGNet(nn.Module):
    """Shared spectral encoder + Bahdanau attention + classifier.

    Input:  (B, C, n_bands)
    Output: (logits (B, n_classes), alpha (B, C))
    """

    def __init__(self, n_bands=5, n_classes=4, d_hidden=64, dropout=0.5):
        super().__init__()
        self.spectral_encoder = nn.Sequential(
            nn.Linear(n_bands, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(d_hidden, d_hidden),
            nn.GELU(),
        )
        self.attn_scorer = nn.Sequential(
            nn.Linear(d_hidden, d_hidden),
            nn.Tanh(),
            nn.Linear(d_hidden, 1, bias=False),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_hidden),
            nn.Linear(d_hidden, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, n_classes),
        )

    def forward(self, x, channel_mask=None):
        h = self.spectral_encoder(x)               # (B, C, d_hidden)
        e = self.attn_scorer(h).squeeze(-1)         # (B, C)
        if channel_mask is not None:
            e = e.masked_fill(channel_mask == 0, float('-inf'))
        alpha = F.softmax(e, dim=-1)                # (B, C)
        context = torch.einsum('bc,bcd->bd', alpha, h)
        logits = self.classifier(context)
        return logits, alpha

    @torch.no_grad()
    def get_channel_importance(self, dataloader, device='cuda'):
        self.eval()
        all_alpha = []
        for x, _ in dataloader:
            x = x.to(device)
            _, alpha = self.forward(x)
            all_alpha.append(alpha.cpu())
        mean_alpha = torch.cat(all_alpha, dim=0).mean(dim=0)
        ranking = mean_alpha.argsort(descending=True)
        return mean_alpha.numpy(), ranking.numpy()


class MLPBaseline(nn.Module):
    """310 -> 4 MLP with BatchNorm + ReLU + Dropout."""

    def __init__(self, input_dim=310, n_classes=4, h1=128, h2=64, dropout=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.BatchNorm1d(h1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h1, h2),
            nn.BatchNorm1d(h2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h2, n_classes),
        )

    def forward(self, x):
        return self.net(x)


class DualAttentionEEGNet(nn.Module):
    """Band attention -> Channel attention -> Classifier.

    Input:  (B, C, n_bands)
    Output: (logits (B, n_classes), channel_alpha (B, C), band_alpha (B, C, n_bands))
    """

    def __init__(self, n_bands=5, n_classes=4, d_band=16, d_chan=64, dropout=0.5):
        super().__init__()
        self.band_embed = nn.Sequential(
            nn.Linear(1, d_band),
            nn.GELU(),
        )
        self.band_attn = nn.Sequential(
            nn.Linear(d_band, d_band // 2),
            nn.Tanh(),
            nn.Linear(d_band // 2, 1, bias=False),
        )
        self.channel_proj = nn.Sequential(
            nn.Linear(d_band, d_chan),
            nn.LayerNorm(d_chan),
            nn.GELU(),
        )
        self.chan_attn = nn.Sequential(
            nn.Linear(d_chan, d_chan // 2),
            nn.Tanh(),
            nn.Linear(d_chan // 2, 1, bias=False),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_chan),
            nn.Linear(d_chan, d_chan),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_chan, n_classes),
        )

    def forward(self, x, channel_mask=None):
        B, C, nb = x.shape
        h_band = self.band_embed(x.unsqueeze(-1))      # (B, C, nb, d_band)
        e_band = self.band_attn(h_band).squeeze(-1)     # (B, C, nb)
        beta = F.softmax(e_band, dim=-1)                 # (B, C, nb)
        h_chan = torch.einsum('bcn,bcnd->bcd', beta, h_band)  # (B, C, d_band)
        h_chan = self.channel_proj(h_chan)                # (B, C, d_chan)
        e_chan = self.chan_attn(h_chan).squeeze(-1)        # (B, C)
        if channel_mask is not None:
            e_chan = e_chan.masked_fill(channel_mask == 0, float('-inf'))
        alpha = F.softmax(e_chan, dim=-1)                # (B, C)
        context = torch.einsum('bc,bcd->bd', alpha, h_chan)
        logits = self.classifier(context)
        return logits, alpha, beta
