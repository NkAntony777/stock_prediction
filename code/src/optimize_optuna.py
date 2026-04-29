"""
Optuna Bayesian Optimization — 全局参数搜索
优化目标: 验证集上的平均周收益率（最大化）
搜索空间: 集成权重、分配策略、Top-K、归一化方式
"""
import sys, os, json, warnings
import numpy as np, pandas as pd
import torch, joblib, lightgbm as lgb
from catboost import CatBoost
import optuna
from optuna.samplers import TPESampler
from tqdm import tqdm

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(__file__))
from config import config
from train import feature_cloums_map, split_train_val_by_last_month
from utils import engineer_features_158plus39
from model import StockTransformer
from model_gru import StockGRU


# ── 全局缓存：预计算所有模型在验证集上的预测 ──
_model_cache = {}


def precompute_predictions():
    """预计算所有模型在每个验证日的预测分数，加速 Optuna 搜索"""
    print("预计算模型预测...")

    data_file = os.path.join(config['data_path'], 'train.csv')
    full_df = pd.read_csv(data_file)
    train_df, val_df, val_start = split_train_val_by_last_month(full_df, config['sequence_length'])

    all_stock_ids = full_df['股票代码'].unique()
    stockid2idx = {sid: idx for idx, sid in enumerate(sorted(all_stock_ids))}

    trans_feats = feature_cloums_map['158+39']
    gbdt_feats = [c for c in trans_feats if c != 'instrument']
    gru_feats = gbdt_feats

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir = config['output_dir']
    fn = engineer_features_158plus39

    # 特征工程（全量数据）
    groups = [g for _, g in full_df.groupby('股票代码', sort=False)]
    processed = [fn(g.copy()) for g in groups]
    data = pd.concat(processed).reset_index(drop=True)
    data['instrument'] = data['股票代码'].map(stockid2idx)
    data = data.dropna(subset=['instrument'])
    data['日期'] = pd.to_datetime(data['日期'])

    # 加载模型
    scaler_t = joblib.load(f'{out_dir}/scaler.pkl')
    scaler_g = joblib.load(f'{out_dir}/scaler_gbdt.pkl')

    data_t = data.copy(); data_t[trans_feats] = data_t[trans_feats].fillna(0)
    data_t[trans_feats] = scaler_t.transform(data_t[trans_feats])

    mt = StockTransformer(input_dim=len(trans_feats), config=config, num_stocks=len(all_stock_ids))
    mt.load_state_dict(torch.load(f'{out_dir}/best_model.pth', map_location=device, weights_only=True))
    mt.to(device).eval()

    lgb_m = lgb.Booster(model_file=f'{out_dir}/lgb_model.txt')
    cat_m = CatBoost(); cat_m.load_model(f'{out_dir}/cat_model.cbm')

    # GRU (optional)
    gru_path = f'{out_dir}/gru_model.pth'
    scaler_gr_path = f'{out_dir}/scaler_gru.pkl'
    use_gru = os.path.exists(gru_path) and os.path.exists(scaler_gr_path)
    if use_gru:
        scaler_gr = joblib.load(scaler_gr_path)
        data_gr = data.copy(); data_gr[gru_feats] = data_gr[gru_feats].fillna(0)
        data_gr[gru_feats] = scaler_gr.transform(data_gr[gru_feats])
        mgr = StockGRU(input_dim=len(gru_feats), hidden_dim=128, num_layers=2, dropout=0.2)
        mgr.load_state_dict(torch.load(gru_path, map_location=device, weights_only=True))
        mgr.to(device).eval()
    else:
        data_gr, mgr = None, None

    # 获取所有验证日期（需要满足 sequence_length 要求）
    all_dates = sorted(data['日期'].unique())
    valid_start_idx = config['sequence_length'] - 1
    valid_dates = all_dates[valid_start_idx:]

    # 只取验证集日期（val_start 之后）
    val_dt = pd.to_datetime(val_start)
    eval_dates = [d for d in valid_dates if d >= val_dt]

    print(f"评估日期: {len(eval_dates)} 天 ({eval_dates[0].date()} ~ {eval_dates[-1].date()})")

    cache = {'dates': eval_dates, 'predictions': []}

    for date in tqdm(eval_dates, desc='预计算'):
        day_data = {}
        stock_ids_avail = sorted(data[data['日期'] <= date]['股票代码'].unique())
        sids_t, t_scores = [], []
        sids_g, l_scores, c_scores = [], [], []
        sids_gr, g_scores = [], []

        # Transformer
        for sid in stock_ids_avail:
            hist = data_t[(data_t['股票代码'] == sid) & (data_t['日期'] <= date)]
            hist = hist.sort_values('日期').tail(config['sequence_length'])
            if len(hist) == config['sequence_length']:
                sids_t.append(sid)
                t_scores.append(hist[trans_feats].values.astype(np.float32))
        if t_scores:
            x = torch.from_numpy(np.array(t_scores)).unsqueeze(0).to(device)
            with torch.no_grad():
                day_data['trans'] = (sids_t, mt(x).squeeze(0).cpu().numpy())

        # GBDT (最新特征)
        day_gbdt = data[(data['日期'] <= date) & (data['股票代码'].isin(stock_ids_avail))]
        latest = day_gbdt.sort_values('日期').groupby('股票代码').tail(1)
        latest_g = latest.copy()
        latest_g[gbdt_feats] = latest_g[gbdt_feats].fillna(0)
        X_g = scaler_g.transform(latest_g[gbdt_feats].values.astype(np.float32))
        day_data['lgb'] = (latest_g['股票代码'].tolist(), lgb_m.predict(X_g))
        day_data['cat'] = (latest_g['股票代码'].tolist(), cat_m.predict(X_g))

        # GRU
        if use_gru:
            sids_gr_tmp, gr_seq = [], []
            for sid in stock_ids_avail:
                hist = data_gr[(data_gr['股票代码'] == sid) & (data_gr['日期'] <= date)]
                hist = hist.sort_values('日期').tail(config['sequence_length'])
                if len(hist) == config['sequence_length']:
                    sids_gr_tmp.append(sid)
                    gr_seq.append(hist[gru_feats].values.astype(np.float32))
            if gr_seq:
                x_gr = torch.from_numpy(np.array(gr_seq)).unsqueeze(0).to(device)
                with torch.no_grad():
                    day_data['gru'] = (sids_gr_tmp, mgr(x_gr).squeeze(0).cpu().numpy())

        # 真实 label: 计算当日 T+1→T+5 的收益率
        labels = {}
        for sid in stock_ids_avail:
            stock_data = data[(data['股票代码'] == sid) & (data['日期'] > date)].sort_values('日期')
            if len(stock_data) >= 5:
                # 找到 T+1 和 T+5 日期
                future_dates = stock_data['日期'].unique()[:5]
                if len(future_dates) >= 5:
                    t1_row = stock_data[stock_data['日期'] == future_dates[0]]
                    t5_row = stock_data[stock_data['日期'] == future_dates[4]]
                    if len(t1_row) > 0 and len(t5_row) > 0:
                        p1 = t1_row['开盘'].iloc[0]
                        p5 = t5_row['开盘'].iloc[0]
                        if p1 > 1e-4:
                            labels[sid] = (p5 - p1) / p1

        day_data['labels'] = labels
        cache['predictions'].append(day_data)

    _model_cache.update(cache)
    print("预计算完成!")
    return cache


def portfolio_return(stocks, weights, labels):
    """计算组合收益率（简化版：label 是 T+1→T+5 收益的代理）"""
    total = 0.0
    for s, w in zip(stocks, weights):
        if s in labels and not np.isnan(labels[s]):
            total += w * labels[s]
    return total


def objective(trial):
    """
    Optuna 目标函数：最大化验证集平均收益率
    """
    # ── 搜索空间 ──
    # 模型启用/权重
    use_trans = trial.suggest_categorical('use_trans', [True, False]) if trial.number > 5 else True
    use_cat = trial.suggest_categorical('use_cat', [True, False]) if trial.number > 5 else True
    use_lgb = trial.suggest_categorical('use_lgb', [True, False])
    use_gru = trial.suggest_categorical('use_gru', [True, False])

    # 原始权重（后归一化）
    w_trans_raw = trial.suggest_float('w_trans', 0.0, 1.0) if use_trans else 0.0
    w_cat_raw = trial.suggest_float('w_cat', 0.0, 1.0) if use_cat else 0.0
    w_lgb_raw = trial.suggest_float('w_lgb', 0.0, 0.5) if use_lgb else 0.0
    w_gru_raw = trial.suggest_float('w_gru', 0.0, 0.3) if use_gru else 0.0

    # 归一化权重
    w_sum = w_trans_raw + w_cat_raw + w_lgb_raw + w_gru_raw
    if w_sum < 1e-12:
        return -1.0
    w_trans = w_trans_raw / w_sum
    w_cat = w_cat_raw / w_sum
    w_lgb = w_lgb_raw / w_sum
    w_gru = w_gru_raw / w_sum

    # 权重分配策略
    weight_strategy = trial.suggest_categorical('weight_strategy',
                                                 ['rank_linear', 'rank_sqrt', 'proportional', 'equal'])
    top_k = trial.suggest_int('top_k', 3, 7)

    # 归一化方式
    norm_method = trial.suggest_categorical('norm_method', ['minmax', 'rank', 'zscore'])

    # ── 在每个验证日上评估 ──
    dates = _model_cache['dates']
    predictions = _model_cache['predictions']
    daily_returns = []

    for day_data in predictions:
        # 融合分数
        score_dict = {}

        def add_scores(key, weight, stocks, scores):
            if weight < 1e-6:
                return
            for i, s in enumerate(stocks):
                score_dict.setdefault(s, {})[key] = float(scores[i])

        if use_trans and 'trans' in day_data:
            add_scores('t', w_trans, *day_data['trans'])
        if use_cat and 'cat' in day_data:
            add_scores('c', w_cat, *day_data['cat'])
        if use_lgb and 'lgb' in day_data:
            add_scores('l', w_lgb, *day_data['lgb'])
        if use_gru and 'gru' in day_data:
            add_scores('g', w_gru, *day_data['gru'])

        if len(score_dict) < top_k:
            continue

        # 归一化
        active_keys = set()
        for v in score_dict.values():
            active_keys.update(v.keys())

        for key in active_keys:
            vals = [v[key] for v in score_dict.values() if key in v]
            if len(vals) < top_k:
                continue
            arr = np.array(vals)

            if norm_method == 'minmax':
                mi, ma = arr.min(), arr.max()
                if ma - mi < 1e-12:
                    continue
                normed = (arr - mi) / (ma - mi)
            elif norm_method == 'rank':
                normed = arr.argsort().argsort() / (len(arr) - 1)
            else:  # zscore
                std = arr.std()
                if std < 1e-12:
                    continue
                normed = (arr - arr.mean()) / std

            for v, nv in zip([v for v in score_dict.values() if key in v], normed):
                v[key + '_n'] = float(nv)

        # 加权分数
        ensemble = {}
        for sid, v in score_dict.items():
            s, w = 0.0, 0.0
            weights_map = {'t': w_trans, 'c': w_cat, 'l': w_lgb, 'g': w_gru}
            for kk, wk in weights_map.items():
                if kk + '_n' in v and wk > 0:
                    s += wk * v[kk + '_n']
                    w += wk
            if w > 0:
                ensemble[sid] = s / w

        # 权重分配
        ranked = sorted(ensemble.items(), key=lambda x: x[1], reverse=True)[:top_k]
        scores_arr = np.array([s for _, s in ranked])

        if weight_strategy == 'equal':
            weights = np.ones(top_k) / top_k
        elif weight_strategy == 'proportional':
            raw = np.maximum(scores_arr, 0)
            weights = raw / (raw.sum() + 1e-12)
        elif weight_strategy == 'rank_sqrt':
            raw = np.sqrt(np.arange(top_k, 0, -1, dtype=float))
            weights = raw / raw.sum()
        else:  # rank_linear
            raw = np.arange(top_k, 0, -1, dtype=float)
            weights = raw / raw.sum()

        stocks = [s for s, _ in ranked]
        labels = day_data.get('labels', {})
        ret = portfolio_return(stocks, weights, labels)
        daily_returns.append(ret)

    if not daily_returns:
        return -1.0

    return np.mean(daily_returns)


def main():
    cache = precompute_predictions()

    # Optuna study
    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=42, n_startup_trials=20),
        study_name='stock_ensemble_optimization',
    )

    print("\n开始 Bayesian 优化 (200 trials)...")
    study.optimize(objective, n_trials=200, show_progress_bar=True)

    print(f"\n{'='*50}")
    print(f"最佳 trial #{study.best_trial.number}")
    print(f"最佳收益率: {study.best_value:.4%}")
    print(f"最佳参数:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    # 保存
    os.makedirs(config['output_dir'], exist_ok=True)
    with open(os.path.join(config['output_dir'], 'best_params.json'), 'w') as f:
        json.dump({
            'params': study.best_params,
            'value': float(study.best_value),
        }, f, indent=2)
    print(f"\n结果已保存: {config['output_dir']}/best_params.json")

    # Top-10 参数组合
    print(f"\nTop-10 参数组合:")
    for i, trial in enumerate(study.best_trials[:10]):
        p = trial.params
        print(f"  #{i+1}: {trial.value:.4%} | "
              f"T={p.get('w_trans',0):.2f} C={p.get('w_cat',0):.2f} "
              f"L={p.get('w_lgb',0):.2f} G={p.get('w_gru',0):.2f} | "
              f"{p.get('weight_strategy','')} top{p.get('top_k',5)} "
              f"norm={p.get('norm_method','')}")

    # 重要性分析
    print(f"\n参数重要性:")
    importances = optuna.importance.get_param_importances(study)
    for k, v in sorted(importances.items(), key=lambda x: x[1], reverse=True):
        print(f"  {k}: {v:.4f}")


if __name__ == '__main__':
    import multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    main()
