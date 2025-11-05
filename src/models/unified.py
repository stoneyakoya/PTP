import math

import torch
import torch.nn as nn
import torch.nn.functional as F

H = 256  # モデル隠れ層次元
E = 32  # 時刻埋め込み次元


class InputProj(nn.Module):
    """
    埋め込み圧縮 + 時刻埋め込み付加
    入力:
      x     (B, L, 1280)
      t_idx (B, T)
    出力:
      z     (B, T, L, H)
      h     (B, L, H)  # オプション参照用
    """

    def __init__(self, d_in=1280, d_model=H, time_emb_dim=E, T=8, max_len=30, **kwargs):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=d_in, out_channels=d_model, kernel_size=5, padding=2)
        self.res_proj = nn.Linear(d_in, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len)
        self.time_emb = nn.Embedding(T, time_emb_dim)
        self.time_proj = nn.Linear(time_emb_dim, d_model)

    def forward(self, x, t_idx):
        B, L, _ = x.shape
        T = t_idx.size(1)

        # x: (B, L, d_in) → (B, d_in, L)
        x_c = F.relu(self.conv(x.transpose(1, 2)))  # (B, d_model, L)
        h = x_c.transpose(1, 2)  # (B, L, d_model)
        h = self.pos_enc(h)

        # 時刻埋め込みを得て H 次元に変換
        t_e = self.time_proj(self.time_emb(t_idx))  # (B, T, H)

        # broadcast 合成
        h = h.unsqueeze(1).expand(-1, T, -1, -1)  # (B, T, L, H)
        t_e = t_e.unsqueeze(2)  # (B, T, 1, H)
        z = h + t_e  # (B, T, L, H)

        return z, h


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()  # (max_len,1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)  # 学習させない

    def forward(self, x):
        # x: (B, L, H) に対して先頭 L 行だけを足し込む
        x = x + self.pe[: x.size(1)]
        return x


class SpatioTemporalBlock(nn.Module):
    """
    2D Self-Attention ブロック:
      1) 残基軸 Self-Attn + FFN + Residual
      2) 時間軸 Self-Attn + FFN + Residual
    """

    def __init__(self, d_model=H, n_head=4, dim_ff=4 * H, dropout=0.1, **kwargs):
        super().__init__()
        # 残基軸
        self.res_attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout)
        self.res_ffn = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(dim_ff, d_model),
            nn.Dropout(dropout),
        )
        # 時間軸
        self.time_attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout)
        self.time_ffn = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(dim_ff, d_model),
            nn.Dropout(dropout),
        )
        # LayerNorm
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)

    def forward(self, x, mask):
        """
        x:    (B, T, L, H)
        mask: (B, L) boolean padding mask
        """
        B, T, L, H = x.shape

        # 1) 残基軸 Self-Attn
        #   (B,T,L,H) → (L, B*T, H)
        x1 = x.reshape(B * T, L, H).transpose(0, 1)
        # mask (B,L) → (B,T,L) → (B*T, L)
        res_mask = mask.unsqueeze(1).expand(-1, T, -1).reshape(B * T, L)
        attn1, _ = self.res_attn(x1, x1, x1, key_padding_mask=~res_mask)
        x1 = self.norm1(x1 + attn1)
        f1 = self.res_ffn(x1.transpose(0, 1))  # (B*T, L, H)
        x1 = self.norm2(x1.transpose(0, 1) + f1)
        x1 = x1.transpose(0, 1).reshape(L, B, T, H).permute(1, 2, 0, 3)

        # 2) 時間軸 Self-Attn
        #   (B,T,L,H) → (T, B*L, H)
        x2 = x1.permute(0, 2, 1, 3).reshape(B * L, T, H).transpose(0, 1)
        attn2, _ = self.time_attn(x2, x2, x2)
        x2 = self.norm3(x2 + attn2)
        f2 = self.time_ffn(x2.transpose(0, 1))  # (B*L, T, H)
        x2 = self.norm4(x2.transpose(0, 1) + f2)
        x2 = x2.transpose(0, 1).reshape(T, B, L, H).permute(1, 0, 2, 3)

        return x2


class MultiTaskDecoder(nn.Module):
    """
    クロスAttention＋2ヘッド出力:
      - LabelLoss予測ヘッド
      - LabelIncorporation予測ヘッド
    """

    def __init__(self, d_model=H, T=8, n_head=4, dropout=0.1, feature=False, **kwargs):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(T, d_model))
        self.cross_attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        self.out_loss = nn.Linear(d_model, 1)
        self.out_inc = nn.Linear(d_model, 1)
        self.out_k = nn.Linear(d_model, 1)
        self.feature = feature

    def forward(self, enc_tokens, mask):
        """
        enc_tokens: (B, T*L, H)
        returns:
          loss_pred (B, T)
          inc_pred  (B, T)
          k_log     (B,)
        """
        # enc_tokens: (B, T*L, H)
        B, TL, H = enc_tokens.shape
        T = self.queries.size(0)
        L = TL // T

        # Queries と KV の準備
        Q = self.queries.unsqueeze(1).expand(-1, B, -1)  # (T, B, H)
        KV = enc_tokens.transpose(0, 1)  # (TL, B, H)

        key_mask = ~mask.unsqueeze(1).expand(-1, T, -1).reshape(B, T * L)

        if not self.feature:
            # causal attn_mask を作る → shape (T, TL)
            # 未来のトークンへのアテンションはマスク
            # token j belongs to past-time floor(j/L)
            j = torch.arange(TL, device=enc_tokens.device)
            token_time = (j // L).unsqueeze(0)  # (1, TL)
            query_time = torch.arange(T, device=enc_tokens.device).unsqueeze(1)  # (T,1)
            # allow only token_time ≤ query_time
            allow = token_time <= query_time  # (T, TL)
            # Use boolean attn_mask (True = mask) to match key_padding_mask type
            attn_mask_bool = ~allow

            attn_out, attn_w = self.cross_attn(
                Q, KV, KV, attn_mask=attn_mask_bool, key_padding_mask=key_mask, need_weights=True, average_attn_weights=True
            )
        else:
            attn_out, attn_w = self.cross_attn(
                Q, KV, KV, attn_mask=None, key_padding_mask=key_mask, need_weights=True, average_attn_weights=True
            )

        attn_out = self.norm(attn_out)

        loss_pred = self.out_loss(attn_out).squeeze(-1).transpose(0, 1)
        inc_pred = self.out_inc(attn_out).squeeze(-1).transpose(0, 1)

        # pool time queries and predict logK
        pooled = attn_out.mean(dim=0)  # (B, H)
        k_log = self.out_k(pooled).squeeze(-1)  # (B,)
        return loss_pred, inc_pred, k_log, attn_w


class STCrossPredictor(nn.Module):
    """
    最終モデル本体:
      InputProj → N×SpatioTemporalBlock → MultiTaskDecoder
    """

    def __init__(self, **cfg):
        super().__init__()
        self.input_proj = InputProj(**cfg)
        self.encoder = nn.ModuleList([SpatioTemporalBlock(**cfg) for _ in range(cfg["num_layers"])])
        self.decoder = MultiTaskDecoder(**cfg)

    def forward(self, embedding, mask, t_idx):
        """
        embedding: (B, L, 1280)
        mask:      (B, L)
        t_idx:     (B, T)
        returns:
          loss_pred: (B, T)
          inc_pred:  (B, T)
          k_log:     (B,)
        """
        z, _ = self.input_proj(embedding, t_idx)
        for blk in self.encoder:
            z = blk(z, mask)  # (B, T, L, H)
        B, T, L, H = z.shape
        tokens = z.reshape(B, T * L, H)  # (B, T*L, H)
        return self.decoder(tokens, mask)
