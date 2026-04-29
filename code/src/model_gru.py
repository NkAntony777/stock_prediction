"""
GRU 排序模型 — 仿 Optiver 冠军方案的集成组件
单股票时序编码 + 注意力聚合 + 排名头部
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class StockGRU(nn.Module):
    """GRU 股票排序模型：每只股票独立编码，输出排序分数"""

    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # 输入投影
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # 双向 GRU
        self.gru = nn.GRU(
            hidden_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
        )

        # 时序注意力
        self.time_attn = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

        # 输出层
        self.output = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, src):
        """
        src: [batch, num_stocks, seq_len, feature_dim]
        returns: [batch, num_stocks]
        """
        batch_size, num_stocks, seq_len, feature_dim = src.shape

        # 合并 batch 和 stock 维度
        x = src.view(batch_size * num_stocks, seq_len, feature_dim)
        x = self.input_proj(x)  # [B*N, L, H]

        # GRU 编码
        gru_out, _ = self.gru(x)  # [B*N, L, H*2]

        # 时序注意力
        attn_scores = self.time_attn(gru_out)  # [B*N, L, 1]
        attn_weights = F.softmax(attn_scores, dim=1)
        context = torch.sum(gru_out * attn_weights, dim=1)  # [B*N, H*2]

        # 输出
        scores = self.output(context)  # [B*N, 1]

        return scores.view(batch_size, num_stocks)
