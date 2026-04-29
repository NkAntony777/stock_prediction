"""
SHAP 特征筛选脚本
从 197 个特征中筛选出对周度选股真正有用的特征。
"""
import sys, os, warnings, json, joblib
import numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(__file__))
from config import config
from train import (
    set_seed, _build_label_and_clean,
    feature_cloums_map, feature_engineer_func_map
)
from utils import engineer_features_158plus39


def load_and_prepare():
    """加载数据，特征工程"""
    data_file = os.path.join(config['data_path'], 'train.csv')
    df = pd.read_csv(data_file)
    df['日期'] = pd.to_datetime(df['日期'])

    all_stock_ids = df['股票代码'].unique()
    stockid2idx = {sid: idx for idx, sid in enumerate(sorted(all_stock_ids))}
    feature_engineer = feature_engineer_func_map[config['feature_num']]

    print("特征工程...")
    groups = [g for _, g in df.groupby('股票代码', sort=False)]
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

    # 标准化
    scaler = StandardScaler()
    data[model_features] = data[model_features].replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=model_features)
    data[model_features] = scaler.fit_transform(data[model_features])

    return data, model_features


def compute_shap_importance(data, features):
    """训练 LGB 并计算 SHAP 特征重要性"""
    print(f"训练 LGB ({len(features)} 特征) 计算 SHAP...")

    X = data[features].values.astype(np.float32)
    y = data['label'].values.astype(np.float32)

    # 快速训练
    lgb_train = lgb.Dataset(X, y)
    params = {
        'objective': 'regression', 'metric': 'rmse',
        'num_leaves': 31, 'learning_rate': 0.05,
        'feature_fraction': 0.8, 'lambda_l1': 1.0,
        'lambda_l2': 1.0, 'verbose': -1, 'num_threads': 4,
        'seed': 42,
    }
    model = lgb.train(params, lgb_train, num_boost_round=200)

    # SHAP (使用 gain importance 近似，速度快)
    # LightGBM 原生 feature_importance 基于 gain/split
    gain_imp = model.feature_importance(importance_type='gain')
    split_imp = model.feature_importance(importance_type='split')

    # 归一化
    gain_imp = gain_imp / gain_imp.sum()
    split_imp = split_imp / split_imp.sum()

    # 综合分数
    combined = 0.7 * gain_imp + 0.3 * split_imp

    importance_df = pd.DataFrame({
        'feature': features,
        'gain': gain_imp,
        'split': split_imp,
        'combined': combined,
    }).sort_values('combined', ascending=False)

    return importance_df


def select_and_test(data, importance_df, top_n=40):
    """选择 top N 特征，快速训练对比"""
    top_features = importance_df.head(top_n)['feature'].tolist()

    print(f"\nTop {top_n} 特征:")
    for i, row in importance_df.head(top_n).iterrows():
        print(f"  {row['feature']}: {row['combined']:.4f}")

    # 快速对比
    X_top = data[top_features].values.astype(np.float32)
    y = data['label'].values.astype(np.float32)

    split = int(len(X_top) * 0.8)
    X_tr, X_va = X_top[:split], X_top[split:]
    y_tr, y_va = y[:split], y[split:]

    params = {
        'objective': 'regression', 'metric': 'rmse',
        'num_leaves': 63, 'learning_rate': 0.04,
        'feature_fraction': 0.8, 'lambda_l1': 2.0, 'lambda_l2': 2.0,
        'verbose': -1, 'num_threads': 4, 'seed': 42,
    }

    lgb_tr = lgb.Dataset(X_tr, y_tr)
    lgb_va = lgb.Dataset(X_va, y_va, reference=lgb_tr)
    model = lgb.train(params, lgb_tr, num_boost_round=2000,
                      valid_sets=[lgb_va],
                      callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])

    preds = model.predict(X_va)
    # 按日期分组的 top-5 收益
    val_data = data.iloc[split:].copy()
    val_data['pred'] = preds
    dates = sorted(val_data['日期'].unique())
    returns = []
    for d in dates:
        day = val_data[val_data['日期'] == d]
        if len(day) < 5:
            continue
        top5 = day.nlargest(5, 'pred')
        returns.append(top5['label'].sum())

    avg_return = np.mean(returns) if returns else 0
    print(f"\nTop {top_n} 特征验证集 avg_top5_return: {avg_return:.4f}")
    return top_features, avg_return


def main():
    set_seed(42)
    data, features = load_and_prepare()
    importance = compute_shap_importance(data, features)

    # 保存完整重要性
    os.makedirs(config['output_dir'], exist_ok=True)
    importance.to_csv(os.path.join(config['output_dir'], 'feature_importance.csv'), index=False)
    print(f"特征重要性已保存: {config['output_dir']}/feature_importance.csv")

    # 测试不同 N
    for n in [20, 30, 40, 50, 70, 100]:
        select_and_test(data, importance, n)

    # 最终推荐: top 40
    top40 = importance.head(40)['feature'].tolist()
    with open(os.path.join(config['output_dir'], 'selected_features.json'), 'w') as f:
        json.dump(top40, f, indent=2)
    print(f"\nTop 40 特征已保存: {config['output_dir']}/selected_features.json")


if __name__ == '__main__':
    import multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    main()
