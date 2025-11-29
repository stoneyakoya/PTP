import math

import torch
import torch.nn as nn


class Transformer_TimeSeq(nn.Module):
    def __init__(
        self,
        input_dim,
        embed_dim=64,
        n_heads=4,
        num_layers=2,
        dropout=0.1,
        activation_func="Sigmoid",
    ):
        super().__init__()

        # 入力の次元圧縮（LSTMと同様にシンプルに）
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(embed_dim, embed_dim),
        )

        # Position Encoding
        self.register_buffer(
            "position_encoding",
            self._create_pe(max_length=8, d_model=embed_dim),  # 8 timesteps
        )

        # Transformerエンコーダ
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=embed_dim * 2,
            dropout=dropout,
            activation="relu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 出力層（LSTMと同様のシンプルな構造）
        self.fc_loss = nn.Linear(embed_dim, 1)
        self.fc_incorporation = nn.Linear(embed_dim, 1)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Activation
        if activation_func == "Sigmoid":
            self.activation = torch.sigmoid
        elif activation_func == "ReLU":
            self.activation = torch.relu
        elif activation_func == "GELU":
            self.activation = nn.functional.gelu
        else:
            raise ValueError("Unsupported activation function")

    def _create_pe(self, max_length, d_model):
        pe = torch.zeros(max_length, d_model)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        return pe

    def forward(self, x, return_embedding=False):
        # x: (batch_size, time_steps, input_dim)
        batch_size, seq_len = x.size(0), x.size(1)

        # 入力の次元圧縮
        x = self.input_projection(x)  # (batch_size, seq_len, embed_dim)

        # Position encodingの追加
        x = x + self.position_encoding[:, :seq_len, :]

        # Transformer処理
        x = x.transpose(0, 1)  # (seq_len, batch_size, embed_dim)
        x = self.transformer(x)
        x = x.transpose(0, 1)  # (batch_size, seq_len, embed_dim)

        x = self.dropout(x)
        if return_embedding:
            # エンコーダの埋め込みをそのまま返す
            return x

        # 出力層
        loss_out = self.fc_loss(x)
        incorporation_out = self.fc_incorporation(x)

        return self.activation(loss_out), self.activation(incorporation_out)


class CustomTransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dropout=0.1,
        dim_feedforward=2048,
    ):
        super(CustomTransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout_ffn = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.activation = nn.ReLU()  # GELU や LeakyReLU なども可

    def forward(self, src):
        # src: (seq_len, batch_size, embed_dim)
        # --- Self-Attention ---
        attn_output, attn_scores = self.self_attn(src, src, src, need_weights=True)
        src2 = self.norm1(src + self.dropout(attn_output))

        # --- Feed Forward ---
        ffn_output = self.linear2(self.dropout_ffn(self.activation(self.linear1(src2))))
        src3 = self.norm2(src2 + self.dropout_ffn(ffn_output))

        return src3, attn_scores


class Transformer_AASeq(nn.Module):
    def __init__(
        self,
        input_dim,
        embed_dim,
        n_heads,
        num_layers,
        dropout=0.5,
        activation_func="Sigmoid",
    ):
        super().__init__()

        # 入力の次元数を埋め込みベクトルに変換する線形層
        self.input_fc_loss = nn.Linear(input_dim, embed_dim)
        self.input_fc_incorporation = nn.Linear(input_dim, embed_dim)

        # Custom Transformer Encoder Layersを使用してアテンションスコアを取得可能にする
        self.encoder_loss_layers = nn.ModuleList(
            [
                CustomTransformerEncoderLayer(embed_dim, n_heads, dropout)
                for _ in range(num_layers)
            ]
        )
        self.encoder_incorp_layers = nn.ModuleList(
            [
                CustomTransformerEncoderLayer(embed_dim, n_heads, dropout)
                for _ in range(num_layers)
            ]
        )

        # 最終的な予測用の全結合層 (lossとincorporation)
        self.fc_loss = nn.Linear(embed_dim, 1)  # Label loss用
        self.fc_incorporation = nn.Linear(embed_dim, 1)  # Label incorporation用

        self.dropout = nn.Dropout(dropout)

        # Activation関数
        if activation_func == "Sigmoid":
            self.activation = torch.sigmoid
        elif activation_func == "ReLU":
            self.activation = torch.relu
        elif activation_func == "GELU":
            self.activation = nn.functional.gelu
        elif activation_func == "LeakyReLU":
            self.activation = nn.functional.leaky_relu
        else:
            raise ValueError(
                "Invalid activation function. Choose 'Sigmoid', 'ReLU', 'GELU', or 'LeakyReLU'."
            )

    def forward(self, x_loss, x_incorporation):
        # Loss入力処理
        x_loss_embed = self.input_fc_loss(x_loss)
        x_loss_embed = self.dropout(x_loss_embed)
        x_loss_embed = x_loss_embed.permute(1, 0, 2)  # (seq_len, batch, embed_dim)

        # Incorporation入力処理
        x_incorp_embed = self.input_fc_incorporation(x_incorporation)
        x_incorp_embed = self.dropout(x_incorp_embed)
        x_incorp_embed = x_incorp_embed.permute(1, 0, 2)

        # Loss系エンコーダ処理
        loss_attn_scores_all = []
        for layer in self.encoder_loss_layers:
            x_loss_embed, loss_attn_scores = layer(x_loss_embed)
            loss_attn_scores_all.append(loss_attn_scores)
        x_loss_embed = x_loss_embed.permute(1, 0, 2)  # (batch, seq_len, embed_dim)

        # Incorporation系エンコーダ処理
        incorp_attn_scores_all = []
        for layer in self.encoder_incorp_layers:
            x_incorp_embed, incorp_attn_scores = layer(x_incorp_embed)
            incorp_attn_scores_all.append(incorp_attn_scores)
        x_incorp_embed = x_incorp_embed.permute(1, 0, 2)

        # プーリング（平均）
        loss_pooled = x_loss_embed.mean(dim=1)
        incorp_pooled = x_incorp_embed.mean(dim=1)

        # タスク出力
        output_loss = self.activation(self.fc_loss(loss_pooled))
        output_incorporation = self.activation(self.fc_incorporation(incorp_pooled))

        return (
            output_loss,
            output_incorporation,
            loss_pooled,
            incorp_pooled,
            loss_attn_scores_all,  # 各レイヤーのアテンションスコアを返す
            incorp_attn_scores_all,
        )


