import torch
import torch.nn as nn


class GRU(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_layers,
        dropout=0.5,
        activation_func="Sigmoid",
        bidirectional=False,
    ):
        super(GRU, self).__init__()

        # 共有GRU
        self.shared_gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.shared_gru.flatten_parameters = lambda: None
        # 全結合層
        if bidirectional:
            gru_output_dim = hidden_dim * 2
        else:
            gru_output_dim = hidden_dim

        self.fc_loss = nn.Linear(gru_output_dim, 1)
        self.fc_incorporation = nn.Linear(gru_output_dim, 1)

        # ドロップアウト
        self.dropout = nn.Dropout(dropout)

        # アクティベーション関数
        if activation_func == "Sigmoid":
            self.activation = torch.sigmoid
        elif activation_func == "ReLU":
            self.activation = torch.relu
        elif activation_func == "GELU":
            self.activation = nn.functional.gelu
        else:
            raise ValueError("Unsupported activation function")

    def forward(self, x):
        """
        x: Tensor of shape (batch, seq_len, input_dim)
        """
        # 共有GRUに入力
        out_shared, _ = self.shared_gru(
            x
        )  # (batch, seq_len, hidden_dim * (2 if bidirectional else 1))

        out_shared = self.dropout(out_shared)

        # Lossタスクの出力
        out_loss = self.fc_loss(out_shared)  # (batch, seq_len, 1)
        output_loss = self.activation(out_loss)  # (batch, seq_len, 1)

        # Incorporationタスクの出力
        out_incorporation = self.fc_incorporation(out_shared)  # (batch, seq_len, 1)
        output_incorporation = self.activation(out_incorporation)  # (batch, seq_len, 1)

        return output_loss, output_incorporation
