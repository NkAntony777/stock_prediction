"""
改进版股票排序模型训练脚本
基于Kaggle冠军方案调研，改进点：
1. LightGBM LambdaRank排序优化
2. Cross-sectional rank特征
3. NDCG@K评估指标
4. 波动率特征增强
"""

import os
import json
import joblib
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from config import config
from train import (
    _preprocess_common, _build_label_and_clean,
    split_train_val_by_last_month, feature_cloums_map,
    feature_engineer_func_map
)

warnings.filterwarnings('ignore')


def add_cross_sectional_rank_features(df, feature_cols):
    """
    添加截面排名特征 - 基于调研的Jane Street冠军方案
    对于每个日期，计算各股票特征的截面排名
    """
    df = df.copy()
    rank_features = []
    
    # 基础价格和成交量特征
    basic_cols = ['开盘', '收盘', '最高', '最低', '成交量', '成交额', '涨跌幅', '换手率']
    existing_basic = [c for c in basic_cols if c in df.columns]
    
    for col in existing_basic:
        rank_col = f'{col}_cs_rank'
        df[rank_col] = df.groupby('日期')[col].rank(pct=True)
        rank_features.append(rank_col)
    
    # 技术指标排名
    tech_cols = ['volatility_10', 'volatility_20', 'return_1', 'return_5', 'return_10',
                 'rsi', 'macd', 'obv', 'volume_change', 'atr_14']
    existing_tech = [c for c in tech_cols if c in df.columns]
    
    for col in existing_tech:
        rank_col = f'{col}_cs_rank'
        df[rank_col] = df.groupby('日期')[col].rank(pct=True)
        rank_features.append(rank_col)
    
    return df, rank_features


def add_volatility_features(df):
    """
    添加波动率相关特征 - 基于Jane Street冠军方案核心发现
    波动率特征是顶级方案的关键
    """
    df = df.copy()
    
    # 已经有的波动率特征
    if 'return_1' in df.columns:
        # 多种窗口的波动率
        for window in [5, 10, 20, 30]:
            col = f'volatility_{window}'
            if col not in df.columns:
                df[col] = df.groupby('股票代码')['return_1'].transform(
                    lambda x: x.rolling(window).std()
                )
        
        # 波动率比率
        if 'volatility_10' in df.columns and 'volatility_20' in df.columns:
            df['vol_ratio_10_20'] = df['volatility_10'] / (df['volatility_20'] + 1e-12)
        
        # 波动率动量（波动率变化）
        if 'volatility_10' in df.columns:
            df['vol_momentum'] = df.groupby('股票代码')['volatility_10'].diff()
        
        # 高波动度标志
        df['high_volatility'] = (df['volatility_10'] > df['volatility_10'].median()).astype(int)
    
    return df


def split_with_embargo(df, sequence_length, embargo_days=5):
    """带Embargo的训练/验证集切分"""
    df = df.copy()
    df['日期'] = pd.to_datetime(df['日期'])
    df = df.sort_values(['日期', '股票代码']).reset_index(drop=True)
    
    last_date = df['日期'].max()
    val_start = (last_date - pd.DateOffset(months=2)).normalize()
    embargo_start = val_start - pd.tseries.offsets.BDay(embargo_days)
    val_context_start = val_start - pd.tseries.offsets.BDay(sequence_length - 1)
    
    train_df = df[df['日期'] < embargo_start].copy()
    val_df = df[df['日期'] >= val_context_start].copy()
    
    print(f"全量数据范围: {df['日期'].min().date()} 到 {last_date.date()}")
    print(f"训练集截止: {train_df['日期'].max().date()} | 验证集起始: {val_start.date()}")
    print(f"验证集实际取数范围(含序列上下文): {val_df['日期'].min().date()} 到 {val_df['日期'].max().date()}")
    
    train_df['日期'] = train_df['日期'].dt.strftime('%Y-%m-%d')
    val_df['日期'] = val_df['日期'].dt.strftime('%Y-%m-%d')
    return train_df, val_df, val_start


def prepare_data():
    """加载数据，做特征工程，返回训练/验证集"""
    data_file = os.path.join(config['data_path'], 'train.csv')
    full_df = pd.read_csv(data_file)
    train_df, val_df, val_start = split_with_embargo(full_df, config['sequence_length'], embargo_days=5)
    
    all_stock_ids = full_df['股票代码'].unique()
    stockid2idx = {sid: idx for idx, sid in enumerate(sorted(all_stock_ids))}
    
    feature_engineer = feature_engineer_func_map[config['feature_num']]
    
    # 训练集特征工程
    print("训练集特征工程...")
    groups = [g for _, g in train_df.groupby('股票代码', sort=False)]
    processed_list = []
    for g in groups:
        processed_list.append(feature_engineer(g.copy()))
    train_processed = pd.concat(processed_list).reset_index(drop=True)
    train_processed['instrument'] = train_processed['股票代码'].map(stockid2idx)
    train_processed = train_processed.dropna(subset=['instrument']).copy()
    train_processed['instrument'] = train_processed['instrument'].astype(np.int64)
    train_processed = _build_label_and_clean(train_processed, drop_small_open=True)
    
    # 验证集特征工程
    print("验证集特征工程...")
    groups_val = [g for _, g in val_df.groupby('股票代码', sort=False)]
    val_list = []
    for g in groups_val:
        val_list.append(feature_engineer(g.copy()))
    val_processed = pd.concat(val_list).reset_index(drop=True)
    val_processed['instrument'] = val_processed['股票代码'].map(stockid2idx)
    val_processed = val_processed.dropna(subset=['instrument']).copy()
    val_processed['instrument'] = val_processed['instrument'].astype(np.int64)
    val_processed = _build_label_and_clean(val_processed, drop_small_open=True)
    
    # 添加截面排名特征
    print("添加截面排名特征...")
    train_processed, rank_features = add_cross_sectional_rank_features(train_processed, feature_cloums_map[config['feature_num']])
    val_processed, _ = add_cross_sectional_rank_features(val_processed, feature_cloums_map[config['feature_num']])
    
    # 添加波动率特征
    print("添加波动率特征...")
    train_processed = add_volatility_features(train_processed)
    val_processed = add_volatility_features(val_processed)
    
    # 特征列
    base_features = [c for c in feature_cloums_map[config['feature_num']] if c not in ('instrument',)]
    model_features = base_features + rank_features
    
    # 添加新波动率特征
    vol_features = ['volatility_5', 'volatility_30', 'vol_ratio_10_20', 'vol_momentum', 'high_volatility']
    for vf in vol_features:
        if vf in train_processed.columns and vf not in model_features:
            model_features.append(vf)
    
    # 标准化
    scaler = StandardScaler()
    train_processed[model_features] = train_processed[model_features].replace([np.inf, -np.inf], np.nan)
    val_processed[model_features] = val_processed[model_features].replace([np.inf, -np.inf], np.nan)
    train_processed = train_processed.dropna(subset=model_features)
    val_processed = val_processed.dropna(subset=model_features)
    train_processed[model_features] = scaler.fit_transform(train_processed[model_features])
    val_processed[model_features] = scaler.transform(val_processed[model_features])
    
    # 保存 scaler
    output_dir = config['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(scaler, os.path.join(output_dir, 'scaler_improved.pkl'))
    
    return train_processed, val_processed, model_features, val_start


def calculate_ndcg(y_true, y_pred, k=5):
    """计算NDCG@K"""
    if len(y_true) < k:
        return 0.0
    
    # 按预测分数排序
    pred_order = np.argsort(y_pred)[::-1][:k]
    true_order = np.argsort(y_true)[::-1][:k]
    
    # DCG
    dcg = 0.0
    for i, idx in enumerate(pred_order):
        dcg += y_true[idx] / np.log2(i + 2)
    
    # IDCG
    idcg = 0.0
    for i, idx in enumerate(true_order):
        idcg += y_true[idx] / np.log2(i + 2)
    
    if idcg == 0:
        return 0.0
    return dcg / idcg


def train_lightgbm_ranker(train_df, val_df, features, output_dir):
    """
    训练 LightGBM LambdaRank 排序模型
    基于调研发现：LightGBM Ranker可达到394%总回报率
    """
    import lightgbm as lgb
    
    # 按日期分组，构建 query group
    train_dates = train_df['日期'].unique()
    train_groups = [train_df[train_df['日期'] == d].shape[0] for d in sorted(train_dates)]
    
    val_dates = val_df['日期'].unique()
    val_groups = [val_df[val_df['日期'] == d].shape[0] for d in sorted(val_dates)]
    
    X_train = train_df[features].values.astype(np.float32)
    y_train = train_df['label'].values.astype(np.float32)
    X_val = val_df[features].values.astype(np.float32)
    y_val = val_df['label'].values.astype(np.float32)
    
    print(f"LightGBM Ranker: train={X_train.shape[0]} samples, val={X_val.shape[0]} samples")
    print(f"Query groups: train={len(train_groups)}, val={len(val_groups)}")
    
    # 创建LambdaRank数据集
    lgb_train = lgb.Dataset(X_train, y_train, group=train_groups)
    lgb_val = lgb.Dataset(X_val, y_val, group=val_groups, reference=lgb_train)
    
    # LambdaRank参数 - 基于调研优化
    params = {
        'objective': 'lambdarank',  # 排序目标
        'metric': 'ndcg',
        'ndcg_eval_at': [5, 10, 20],
        'boosting_type': 'gbdt',
        'num_leaves': 63,
        'learning_rate': 0.05,  # 稍高学习率
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_data_in_leaf': 30,
        'lambda_l1': 1.0,
        'lambda_l2': 1.0,
        'verbose': 1,
        'num_threads': 4,
        'seed': 42,
        'label_gain': [0, 1, 2, 3, 4, 5],  # 收益等级增益
    }
    
    model = lgb.train(
        params,
        lgb_train,
        num_boost_round=2000,
        valid_sets=[lgb_train, lgb_val],
        valid_names=['train', 'val'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=100),
            lgb.log_evaluation(period=50),
        ],
    )
    
    model_path = os.path.join(output_dir, 'lgb_ranker_model.txt')
    model.save_model(model_path)
    print(f"LightGBM Ranker 模型已保存: {model_path}")
    
    # 评估
    _eval_ranker(model, X_val, y_val, val_df, 'LightGBM Ranker', output_dir)
    return model


def _eval_ranker(model, X, y, df, name, output_dir):
    """评估排序模型 - 计算NDCG@K和Top-K收益率"""
    dates = sorted(df['日期'].unique())
    ndcg_scores = []
    pred_returns = []
    max_returns = []
    random_returns = []
    
    for d in dates:
        mask = df['日期'] == d
        X_d = X[mask.values]
        y_d = y[mask.values]
        if len(X_d) < 5:
            continue
        
        scores = model.predict(X_d)
        
        # 计算NDCG@5
        ndcg = calculate_ndcg(y_d, scores, k=5)
        ndcg_scores.append(ndcg)
        
        # Top-5收益率
        top5_idx = np.argsort(scores)[-5:]
        pred_returns.append(y_d[top5_idx].sum())
        
        # 理论最大Top-5收益率
        true_top5_idx = np.argsort(y_d)[-5:]
        max_returns.append(y_d[true_top5_idx].sum())
        
        # 随机选择Top-5期望收益
        random_returns.append(5 * np.mean(y_d))
    
    avg_ndcg = np.mean(ndcg_scores)
    avg_pred = np.mean(pred_returns)
    avg_max = np.mean(max_returns)
    avg_random = np.mean(random_returns)
    
    # Final score（类似比赛评分）
    final_score = (avg_pred - avg_random) / (avg_max - avg_random + 1e-12)
    
    print(f"\n{name} 评估结果:")
    print(f"  NDCG@5: {avg_ndcg:.4f}")
    print(f"  Pred Top5 Return Sum: {avg_pred:.4f}")
    print(f"  Max Top5 Return Sum: {avg_max:.4f}")
    print(f"  Random Top5 Return Sum: {avg_random:.4f}")
    print(f"  Final Score: {final_score:.4f}")
    
    # 保存评估结果
    eval_results = {
        'model': name,
        'ndcg@5': avg_ndcg,
        'pred_top5_return': avg_pred,
        'max_top5_return': avg_max,
        'random_top5_return': avg_random,
        'final_score': final_score
    }
    
    eval_path = os.path.join(output_dir, f'{name.replace(" ", "_").lower()}_eval.json')
    with open(eval_path, 'w') as f:
        json.dump(eval_results, f, indent=2)
    
    return eval_results


def main():
    """主训练流程"""
    set_seed_42 = lambda seed=42: np.random.seed(seed) if hasattr(np.random, 'seed') else None
    try:
        np.random.seed(42)
    except:
        pass
    
    output_dir = config['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存配置
    with open(os.path.join(output_dir, 'config_improved.json'), 'w') as f:
        json.dump(config, f, indent=4, ensure_ascii=False)
    
    train_df, val_df, features, val_start = prepare_data()
    
    print(f"\n总特征数: {len(features)}")
    print(f"训练集样本: {len(train_df)}, 验证集样本: {len(val_df)}")
    
    # 训练 LightGBM LambdaRank
    print("\n" + "=" * 50)
    print("训练 LightGBM LambdaRank...")
    print("=" * 50)
    model = train_lightgbm_ranker(train_df, val_df, features, output_dir)
    
    print("\n" + "=" * 50)
    print("改进版模型训练完成!")
    print(f"输出目录: {output_dir}")
    print("=" * 50)


if __name__ == '__main__':
    import multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    main()
