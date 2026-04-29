"""
CatBoost + LightGBM 排序模型训练
三模型集成的树模型部分（仿 Optiver 冠军方案）
"""
import os, json, joblib, warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from config import config
from train import (
    set_seed, _preprocess_common, _build_label_and_clean,
    split_train_val_by_last_month, feature_cloums_map,
    feature_engineer_func_map
)

warnings.filterwarnings('ignore')

# ── 数据准备（复用 baseline 的预处理流程） ──
def split_with_embargo(df, sequence_length, embargo_days=5):
    """
    Purged K-Fold 风格的训练/验证集切分。
    在训练/验证边界插入 embargo 期，防止序列相关性泄露。
    """
    df = df.copy()
    df['日期'] = pd.to_datetime(df['日期'])
    df = df.sort_values(['日期', '股票代码']).reset_index(drop=True)

    last_date = df['日期'].max()
    val_start = (last_date - pd.DateOffset(months=2)).normalize()

    # Embargo: 验证集起始日前推 embargo_days 个交易日作为训练截止
    # 避免因相邻日序列相关性导致的信息泄露
    val_dates = sorted(df['日期'].unique())
    val_dates_series = pd.Series(val_dates)
    embargo_start = val_start - pd.tseries.offsets.BDay(embargo_days)

    # 验证集需要前 sequence_length-1 个交易日作为序列上下文
    val_context_start = val_start - pd.tseries.offsets.BDay(sequence_length - 1)

    train_df = df[df['日期'] < embargo_start].copy()
    val_df = df[df['日期'] >= val_context_start].copy()

    print(f"全量数据范围: {df['日期'].min().date()} 到 {last_date.date()}")
    print(f"训练集截止: {train_df['日期'].max().date()} | 验证集起始: {val_start.date()}")
    print(f"Embargo 期: {(pd.to_datetime(train_df['日期'].max()) + pd.Timedelta(days=1)).date()} ~ {(pd.to_datetime(val_start) - pd.Timedelta(days=1)).date()}")
    print(f"验证集实际取数范围(含序列上下文): {val_df['日期'].min().date()} 到 {val_df['日期'].max().date()}")

    train_df['日期'] = train_df['日期'].dt.strftime('%Y-%m-%d')
    val_df['日期'] = val_df['日期'].dt.strftime('%Y-%m-%d')
    return train_df, val_df, val_start


def prepare_data():
    """加载数据，做特征工程，返回训练/验证集（DataFrame格式，不做序列化）"""
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

    # 验证集特征工程（串行处理防OOM）
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

    # 特征列
    feature_cols = [c for c in feature_cloums_map[config['feature_num']] if c not in ('instrument',)]
    model_features = [c for c in feature_cols if c not in ('instrument',)]

    # 标准化
    scaler = StandardScaler()
    train_processed[model_features] = train_processed[model_features].replace([np.inf, -np.inf], np.nan)
    val_processed[model_features] = val_processed[model_features].replace([np.inf, -np.inf], np.nan)
    train_processed = train_processed.dropna(subset=model_features)
    val_processed = val_processed.dropna(subset=model_features)
    train_processed[model_features] = scaler.fit_transform(train_processed[model_features])
    val_processed[model_features] = scaler.transform(val_processed[model_features])

    # 保存 scaler
    os.makedirs(config['output_dir'], exist_ok=True)
    joblib.dump(scaler, os.path.join(config['output_dir'], 'scaler_gbdt.pkl'))

    return train_processed, val_processed, model_features, val_start


def train_lightgbm(train_df, val_df, features, output_dir):
    """训练 LightGBM 排序模型"""
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

    # 回归目标（RMSE）：LambdaRank 在验证集上提升但测试集不泛化
    # 保留 embargo 切分防泄露
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)

    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 63,
        'learning_rate': 0.04,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_data_in_leaf': 30,
        'lambda_l1': 2.0,
        'lambda_l2': 2.0,
        'verbose': 1,
        'num_threads': 4,
        'seed': 42,
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

    # LambdaRank 预测的是 relevance score (float)，可以直接排序
    # _eval_top5 中使用 argsort 即可

    model_path = os.path.join(output_dir, 'lgb_model.txt')
    model.save_model(model_path)
    print(f"LightGBM 模型已保存: {model_path}")

    # 计算验证集 top-5 收益
    _eval_top5(model, X_val, y_val, val_df, 'LightGBM')
    return model


def train_catboost(train_df, val_df, features, output_dir):
    """训练 CatBoost 排序模型"""
    from catboost import CatBoost, Pool

    # 按日期分组，构建 query group
    train_dates = sorted(train_df['日期'].unique())
    train_groups = [train_df[train_df['日期'] == d].shape[0] for d in train_dates]
    val_dates = sorted(val_df['日期'].unique())
    val_groups = [val_df[val_df['日期'] == d].shape[0] for d in val_dates]

    X_train = train_df[features].values.astype(np.float32)
    y_train = train_df['label'].values.astype(np.float32)
    X_val = val_df[features].values.astype(np.float32)
    y_val = val_df['label'].values.astype(np.float32)

    # CatBoost 需要 group_id 列
    train_group_id = np.repeat(np.arange(len(train_groups)), train_groups)
    val_group_id = np.repeat(np.arange(len(val_groups)), val_groups)

    print(f"CatBoost: train={X_train.shape[0]} samples, val={X_val.shape[0]} samples")

    train_pool = Pool(X_train, y_train, group_id=train_group_id)
    val_pool = Pool(X_val, y_val, group_id=val_group_id)

    model = CatBoost(
        {
            'loss_function': 'RMSE',
            'custom_metric': ['MAE'],
            'iterations': 2000,
            'learning_rate': 0.04,
            'depth': 6,
            'l2_leaf_reg': 5.0,
            'random_strength': 1.0,
            'border_count': 128,
            'random_seed': 42,
            'verbose': 100,
            'thread_count': 4,
            'task_type': 'CPU',
        }
    )

    model.fit(
        train_pool,
        eval_set=val_pool,
        early_stopping_rounds=100,
    )

    model_path = os.path.join(output_dir, 'cat_model.cbm')
    model.save_model(model_path)
    print(f"CatBoost 模型已保存: {model_path}")

    _eval_top5(model, X_val, y_val, val_df, 'CatBoost')
    return model


def _eval_top5(model, X, y, df, name):
    """评估 top-5 收益率（模拟比赛打分）"""
    dates = sorted(df['日期'].unique())
    pred_returns = []
    max_returns = []

    for d in dates:
        mask = df['日期'] == d
        X_d = X[mask.values]
        y_d = y[mask.values]
        if len(X_d) < 5:
            continue

        if hasattr(model, 'predict'):
            scores = model.predict(X_d)
        else:
            scores = model.predict(X_d).flatten()

        top5_idx = np.argsort(scores)[-5:]
        pred_returns.append(y_d[top5_idx].sum())
        true_top5_idx = np.argsort(y_d)[-5:]
        max_returns.append(y_d[true_top5_idx].sum())

    avg_pred = np.mean(pred_returns)
    avg_max = np.mean(max_returns)
    avg_random = 5 * np.mean(y)
    final_score = (avg_pred - avg_random) / (avg_max - avg_random + 1e-12)

    print(f"  {name} pred_return_sum: {avg_pred:.4f}, max_return_sum: {avg_max:.4f}")
    print(f"  {name} final_score: {final_score:.4f}")


def main():
    set_seed(42)
    output_dir = config['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4, ensure_ascii=False)

    train_df, val_df, features, val_start = prepare_data()

    # ── 仅用非序列特征（去掉需要序列上下文的特征，这里直接用全部特征即可）──
    # 树模型不需要序列，每条样本就是 (日期, 股票)  → label
    # 但当前特征包含了滚动窗口值，可以直接用

    print(f"\n总特征数: {len(features)}")

    # 训练 LightGBM
    print("\n" + "=" * 50)
    print("训练 LightGBM...")
    print("=" * 50)
    lgb_model = train_lightgbm(train_df, val_df, features, output_dir)

    # 训练 CatBoost
    print("\n" + "=" * 50)
    print("训练 CatBoost...")
    print("=" * 50)
    cat_model = train_catboost(train_df, val_df, features, output_dir)

    print("\n" + "=" * 50)
    print("GBDT 模型训练完成!")
    print(f"输出目录: {output_dir}")
    print("=" * 50)


if __name__ == '__main__':
    import multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    main()
