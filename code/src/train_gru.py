"""
GRU 排序模型训练脚本
仿 Optiver 冠军方案: GRU + Transformer + CatBoost 三模型集成
"""
import os, json, joblib, random
import numpy as np, pandas as pd
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from config import config
from model_gru import StockGRU
from train import (
    set_seed, _build_label_and_clean, split_train_val_by_last_month,
    feature_cloums_map, feature_engineer_func_map,
    RankingDataset, collate_fn, calculate_ranking_metrics
)
import warnings
warnings.filterwarnings('ignore')


def preprocess_data(df, stockid2idx, feature_engineer, is_train=True):
    """特征工程 + 标签构建"""
    groups = [g for _, g in df.groupby('股票代码', sort=False)]
    processed = []
    for g in groups:
        processed.append(feature_engineer(g.copy()))
    data = pd.concat(processed).reset_index(drop=True)
    data['instrument'] = data['股票代码'].map(stockid2idx)
    data = data.dropna(subset=['instrument']).copy()
    data['instrument'] = data['instrument'].astype(np.int64)

    # 排序 + 清洗
    data = data.sort_values(['股票代码', '日期']).reset_index(drop=True)
    data['open_t1'] = data.groupby('股票代码')['开盘'].shift(-1)
    data['open_t5'] = data.groupby('股票代码')['开盘'].shift(-5)
    if is_train:
        data = data[data['open_t1'] > 1e-4]
    data['label'] = (data['open_t5'] - data['open_t1']) / (data['open_t1'] + 1e-12)
    data = data.dropna(subset=['label'])
    data.drop(columns=['open_t1', 'open_t5'], inplace=True)

    return data


class WeightedMSELoss(nn.Module):
    """带 top-k 加权的 MSE 损失"""
    def __init__(self, k=5, topk_weight=3.0):
        super().__init__()
        self.k = k
        self.topk_weight = topk_weight

    def forward(self, preds, targets):
        # preds/targets: [batch, num_stocks]
        loss = (preds - targets) ** 2
        # 对真实 top-k 样本加权
        weights = torch.ones_like(targets)
        for i in range(targets.size(0)):
            _, top_idx = torch.topk(targets[i], min(self.k, targets.size(1)))
            weights[i, top_idx] = self.topk_weight
        return (loss * weights).mean()


def create_sequences(data, features, sequence_length):
    """创建 GRU 排序数据集（复用 baseline 的向量化逻辑）"""
    from utils import create_ranking_dataset_vectorized
    return create_ranking_dataset_vectorized(
        data, features, sequence_length,
        ranking_data_path=None
    )


def main():
    set_seed(42)
    output_dir = config['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    data_file = os.path.join(config['data_path'], 'train.csv')
    full_df = pd.read_csv(data_file)
    train_df, val_df, val_start = split_train_val_by_last_month(full_df, config['sequence_length'])

    all_stock_ids = full_df['股票代码'].unique()
    stockid2idx = {sid: idx for idx, sid in enumerate(sorted(all_stock_ids))}

    feature_engineer = feature_engineer_func_map[config['feature_num']]
    feature_cols = feature_cloums_map[config['feature_num']]
    model_features = [c for c in feature_cols if c not in ('instrument',)]  # GRU: 196维

    print("特征工程...")
    train_data = preprocess_data(train_df, stockid2idx, feature_engineer, is_train=True)
    val_data = preprocess_data(val_df, stockid2idx, feature_engineer, is_train=True)

    # 标准化
    scaler = StandardScaler()
    train_data[model_features] = train_data[model_features].replace([np.inf, -np.inf], np.nan)
    val_data[model_features] = val_data[model_features].replace([np.inf, -np.inf], np.nan)
    train_data = train_data.dropna(subset=model_features)
    val_data = val_data.dropna(subset=model_features)
    train_data[model_features] = scaler.fit_transform(train_data[model_features])
    val_data[model_features] = scaler.transform(val_data[model_features])
    joblib.dump(scaler, os.path.join(output_dir, 'scaler_gru.pkl'))

    # 创建序列数据集 (196维，不含 instrument)
    print("创建序列数据集...")
    train_seq, train_tgt, train_rel, train_idx = create_sequences(
        train_data, model_features, config['sequence_length']
    )
    val_seq, val_tgt, val_rel, val_idx = create_sequences(
        val_data, model_features, config['sequence_length'],
    )
    # 过滤验证集（只保留 val_start 之后的窗口）
    if val_start is not None:
        valid = [i for i, sid_list in enumerate(val_idx)
                 if len(sid_list) > 0]
        val_seq = [val_seq[i] for i in valid]
        val_tgt = [val_tgt[i] for i in valid]
        val_rel = [val_rel[i] for i in valid]
        val_idx = [val_idx[i] for i in valid]

    print(f"Train: {len(train_seq)} days, Val: {len(val_seq)} days")

    train_ds = RankingDataset(train_seq, train_tgt, train_rel, train_idx)
    val_ds = RankingDataset(val_seq, val_tgt, val_rel, val_idx)
    train_loader = DataLoader(train_ds, batch_size=config['batch_size'],
                              shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=config['batch_size'],
                            shuffle=False, collate_fn=collate_fn, num_workers=0)

    # 设备
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # 模型
    model = StockGRU(
        input_dim=len(model_features),
        hidden_dim=128,
        num_layers=2,
        dropout=0.2,
    ).to(device)
    print(f"GRU 参数: {sum(p.numel() for p in model.parameters()):,}")

    criterion = WeightedMSELoss(k=5, topk_weight=3.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

    best_score = -float('inf')
    patience = 10
    no_improve = 0

    for epoch in range(30):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}', leave=False):
            seq = batch['sequences'].to(device)
            tgt = batch['targets'].to(device)
            mask = batch['masks'].to(device)

            pred = model(seq)
            pred_masked = pred * mask + (1 - mask) * (-1e9)
            tgt_masked = tgt * mask

            loss = criterion(pred_masked, tgt_masked)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        # 验证
        model.eval()
        val_metrics = {}
        n_val = 0
        with torch.no_grad():
            for batch in val_loader:
                seq = batch['sequences'].to(device)
                tgt = batch['targets'].to(device)
                mask = batch['masks'].to(device)
                pred = model(seq)
                pred_masked = pred * mask + (1 - mask) * (-1e9)
                tgt_masked = tgt * mask
                m = calculate_ranking_metrics(pred_masked, tgt_masked, mask, k=5)
                for k, v in m.items():
                    val_metrics[k] = val_metrics.get(k, 0) + v
                n_val += 1

        for k in val_metrics:
            val_metrics[k] /= max(n_val, 1)

        fs = val_metrics.get('final_score', 0)
        print(f"Epoch {epoch+1}: loss={total_loss/len(train_loader):.4f}, "
              f"val_score={fs:.4f}, pred_ret={val_metrics.get('pred_return_sum',0):.4f}")

        if fs > best_score:
            best_score = fs
            no_improve = 0
            torch.save(model.state_dict(), os.path.join(output_dir, 'gru_model.pth'))
            print(f"  保存最佳模型, score={fs:.4f}")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  早停 at epoch {epoch+1}")
                break

    print(f"\nGRU 训练完成! best_score={best_score:.4f}")


if __name__ == '__main__':
    import multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    main()
