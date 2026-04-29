"""
滚动训练脚本 (Optiver 冠军核心技巧)
支持：固定窗口 / 扩展窗口 / 增量微调三种模式
"""
import os, json, joblib, warnings, copy, argparse
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from config import config
from train import (
    set_seed, _build_label_and_clean, split_train_val_by_last_month,
    feature_cloums_map, feature_engineer_func_map
)
warnings.filterwarnings('ignore')


def prepare_window_data(df, start_date, end_date, stockid2idx, feature_engineer):
    """准备单个窗口的数据"""
    window = df[(df['日期'] >= start_date) & (df['日期'] <= end_date)].copy()
    if len(window) == 0:
        return None, None

    groups = [g for _, g in window.groupby('股票代码', sort=False)]
    processed = []
    for g in groups:
        processed.append(feature_engineer(g.copy()))
    data = pd.concat(processed).reset_index(drop=True)
    data['instrument'] = data['股票代码'].map(stockid2idx)
    data = data.dropna(subset=['instrument']).copy()
    data['instrument'] = data['instrument'].astype(np.int64)
    data = _build_label_and_clean(data, drop_small_open=True)

    feature_cols = feature_cloums_map[config['feature_num']]
    model_features = [c for c in feature_cols if c not in ('instrument',)]

    return data, model_features


def train_window_models(train_data, features, output_dir, window_label):
    """训练单个窗口的 LightGBM + CatBoost"""
    import lightgbm as lgb
    from catboost import CatBoost

    X = train_data[features].values.astype(np.float32)
    y = train_data['label'].values.astype(np.float32)

    # ── LightGBM ──
    lgb_train = lgb.Dataset(X, y)
    params = {
        'objective': 'regression', 'metric': 'rmse',
        'boosting_type': 'gbdt', 'num_leaves': 63,
        'learning_rate': 0.04, 'feature_fraction': 0.8,
        'bagging_fraction': 0.8, 'bagging_freq': 5,
        'min_data_in_leaf': 30, 'lambda_l1': 2.0, 'lambda_l2': 2.0,
        'verbose': -1, 'num_threads': 4, 'seed': 42,
    }
    # 使用最后 10% 数据做验证集防止过拟合
    split_idx = int(len(X) * 0.9)
    lgb_train_data = lgb.Dataset(X[:split_idx], y[:split_idx])
    lgb_val_data = lgb.Dataset(X[split_idx:], y[split_idx:], reference=lgb_train_data)

    lgb_model = lgb.train(
        params, lgb_train_data,
        num_boost_round=500,
        valid_sets=[lgb_val_data],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
    )
    lgb_path = os.path.join(output_dir, f'lgb_{window_label}.txt')
    lgb_model.save_model(lgb_path)
    n_trees = lgb_model.num_trees()
    print(f"  LGB saved: {lgb_path} (trees={n_trees})")

    # ── CatBoost ──
    cat_model = CatBoost({
        'loss_function': 'RMSE', 'iterations': 500,
        'learning_rate': 0.04, 'depth': 6, 'l2_leaf_reg': 5.0,
        'random_seed': 42, 'verbose': 0, 'thread_count': 4,
        'early_stopping_rounds': 50,
    })
    cat_model.fit(X, y, eval_set=(X[int(len(X)*0.9):], y[int(len(y)*0.9):]))
    cat_path = os.path.join(output_dir, f'cat_{window_label}.cbm')
    cat_model.save_model(cat_path)
    print(f"  CatBoost saved: {cat_path} (iters={cat_model.tree_count_})")

    return lgb_model, cat_model


def rolling_train(mode='fixed', window_months=6, step_months=1):
    """
    滚动训练主函数
    mode: 'fixed' — 固定窗口 (最近 N 个月)
          'expanding' — 扩展窗口 (从头到最近)
          'increment' — 增量微调 (加载旧模型 + 新数据续训)
    """
    set_seed(42)
    output_dir = config['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    data_file = os.path.join(config['data_path'], 'train.csv')
    full_df = pd.read_csv(data_file)
    full_df['日期'] = pd.to_datetime(full_df['日期'])

    all_stock_ids = full_df['股票代码'].unique()
    stockid2idx = {sid: idx for idx, sid in enumerate(sorted(all_stock_ids))}
    feature_engineer = feature_engineer_func_map[config['feature_num']]

    last_date = full_df['日期'].max()
    print(f"数据范围: {full_df['日期'].min().date()} ~ {last_date.date()}")
    print(f"滚动模式: {mode}, 窗口={window_months}月\n")

    if mode == 'fixed':
        # 固定窗口：只用最近 N 个月
        start_date = last_date - pd.DateOffset(months=window_months)
        print(f"训练窗口: {start_date.date()} ~ {last_date.date()}")

        train_data, features = prepare_window_data(
            full_df, start_date, last_date, stockid2idx, feature_engineer
        )

        # 标准化
        model_cols = [c for c in features if c in train_data.columns]
        scaler = StandardScaler()
        train_data[model_cols] = train_data[model_cols].replace([np.inf, -np.inf], np.nan)
        train_data = train_data.dropna(subset=model_cols)
        train_data[model_cols] = scaler.fit_transform(train_data[model_cols])

        joblib.dump(scaler, os.path.join(output_dir, 'scaler_gbdt.pkl'))
        print(f"训练样本: {len(train_data)}")

        lgb_m, cat_m = train_window_models(
            train_data, model_cols, output_dir, f'rolling_{window_months}m'
        )

    elif mode == 'expanding':
        # 扩展窗口：从起始到最近，每隔 step_months 保存一个模型
        all_dates = sorted(full_df['日期'].unique())
        start_date = pd.Timestamp(all_dates[0])
        current_end = last_date

        # 只保存最终模型（扩展窗口到最后）
        train_data, features = prepare_window_data(
            full_df, start_date, current_end, stockid2idx, feature_engineer
        )
        model_cols = [c for c in features if c in train_data.columns]
        scaler = StandardScaler()
        train_data[model_cols] = train_data[model_cols].replace([np.inf, -np.inf], np.nan)
        train_data = train_data.dropna(subset=model_cols)
        train_data[model_cols] = scaler.fit_transform(train_data[model_cols])
        joblib.dump(scaler, os.path.join(output_dir, 'scaler_gbdt.pkl'))
        print(f"扩展窗口训练样本: {len(train_data)}")

        lgb_m, cat_m = train_window_models(
            train_data, model_cols, output_dir, 'expanding_full'
        )

    print(f"\n滚动训练完成! 输出目录: {output_dir}")
    return lgb_m, cat_m


if __name__ == '__main__':
    import multiprocessing as mp
    mp.set_start_method('spawn', force=True)

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='fixed', choices=['fixed', 'expanding'])
    parser.add_argument('--window', type=int, default=6, help='窗口月数')
    args = parser.parse_args()

    rolling_train(mode=args.mode, window_months=args.window)
