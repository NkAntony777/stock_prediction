"""
集成预测脚本 — Transformer + LightGBM + CatBoost 加权融合
仿 Optiver 冠军方案: 0.5 * CatBoost + 0.3 * Transformer + 0.2 * LightGBM
"""
import os, joblib, json
import numpy as np
import pandas as pd
import torch
from config import config
from model import StockTransformer
from utils import engineer_features_39, engineer_features_158plus39


feature_cloums_map = {
    '39': [
        'instrument', '开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌额', '换手率', '涨跌幅',
        'sma_5', 'sma_20', 'ema_12', 'ema_26', 'rsi', 'macd', 'macd_signal', 'volume_change', 'obv',
        'volume_ma_5', 'volume_ma_20', 'volume_ratio', 'kdj_k', 'kdj_d', 'kdj_j', 'boll_mid', 'boll_std',
        'atr_14', 'ema_60', 'volatility_10', 'volatility_20', 'return_1', 'return_5', 'return_10',
        'high_low_spread', 'open_close_spread', 'high_close_spread', 'low_close_spread'
    ],
    '158+39': [
        'instrument', '开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌额', '换手率', '涨跌幅',
        'KMID', 'KLEN', 'KMID2', 'KUP', 'KUP2', 'KLOW', 'KLOW2', 'KSFT', 'KSFT2', 'OPEN0', 'HIGH0', 'LOW0',
        'VWAP0', 'ROC5', 'ROC10', 'ROC20', 'ROC30', 'ROC60', 'MA5', 'MA10', 'MA20', 'MA30', 'MA60', 'STD5',
        'STD10', 'STD20', 'STD30', 'STD60', 'BETA5', 'BETA10', 'BETA20', 'BETA30', 'BETA60', 'RSQR5', 'RSQR10',
        'RSQR20', 'RSQR30', 'RSQR60', 'RESI5', 'RESI10', 'RESI20', 'RESI30', 'RESI60', 'MAX5', 'MAX10', 'MAX20',
        'MAX30', 'MAX60', 'MIN5', 'MIN10', 'MIN20', 'MIN30', 'MIN60', 'QTLU5', 'QTLU10', 'QTLU20', 'QTLU30',
        'QTLU60', 'QTLD5', 'QTLD10', 'QTLD20', 'QTLD30', 'QTLD60', 'RANK5', 'RANK10', 'RANK20', 'RANK30',
        'RANK60', 'RSV5', 'RSV10', 'RSV20', 'RSV30', 'RSV60', 'IMAX5', 'IMAX10', 'IMAX20', 'IMAX30', 'IMAX60',
        'IMIN5', 'IMIN10', 'IMIN20', 'IMIN30', 'IMIN60', 'IMXD5', 'IMXD10', 'IMXD20', 'IMXD30', 'IMXD60',
        'CORR5', 'CORR10', 'CORR20', 'CORR30', 'CORR60', 'CORD5', 'CORD10', 'CORD20', 'CORD30', 'CORD60',
        'CNTP5', 'CNTP10', 'CNTP20', 'CNTP30', 'CNTP60', 'CNTN5', 'CNTN10', 'CNTN20', 'CNTN30', 'CNTN60',
        'CNTD5', 'CNTD10', 'CNTD20', 'CNTD30', 'CNTD60', 'SUMP5', 'SUMP10', 'SUMP20', 'SUMP30', 'SUMP60',
        'SUMN5', 'SUMN10', 'SUMN20', 'SUMN30', 'SUMN60', 'SUMD5', 'SUMD10', 'SUMD20', 'SUMD30', 'SUMD60',
        'VMA5', 'VMA10', 'VMA20', 'VMA30', 'VMA60', 'VSTD5', 'VSTD10', 'VSTD20', 'VSTD30', 'VSTD60', 'WVMA5',
        'WVMA10', 'WVMA20', 'WVMA30', 'WVMA60', 'VSUMP5', 'VSUMP10', 'VSUMP20', 'VSUMP30', 'VSUMP60', 'VSUMN5',
        'VSUMN10', 'VSUMN20', 'VSUMN30', 'VSUMN60', 'VSUMD5', 'VSUMD10', 'VSUMD20', 'VSUMD30', 'VSUMD60',
        'sma_5', 'sma_20', 'ema_12', 'ema_26', 'rsi', 'macd', 'macd_signal', 'volume_change', 'obv',
        'volume_ma_5', 'volume_ma_20', 'volume_ratio', 'kdj_k', 'kdj_d', 'kdj_j', 'boll_mid', 'boll_std',
        'atr_14', 'ema_60', 'volatility_10', 'volatility_20', 'return_1', 'return_5', 'return_10',
        'high_low_spread', 'open_close_spread', 'high_close_spread', 'low_close_spread'
    ]
}

feature_engineer_func_map = {
    '39': engineer_features_39,
    '158+39': engineer_features_158plus39,
}


def _preprocess_parallel(data_df, stockid2idx):
    """并行特征工程（同 baseline）"""
    import multiprocessing as mp

    fn = feature_engineer_func_map[config['feature_num']]
    groups = [g for _, g in data_df.groupby('股票代码', sort=False)]
    num_processes = min(2, mp.cpu_count())

    with mp.Pool(processes=num_processes) as pool:
        processed_list = pool.map(fn, groups)
    data = pd.concat(processed_list).reset_index(drop=True)
    data['instrument'] = data['股票代码'].map(stockid2idx)
    data = data.dropna(subset=['instrument']).copy()
    data['instrument'] = data['instrument'].astype(np.int64)
    return data


def get_transformer_scores(data, features, model, scaler, stock_ids, device):
    """Transformer 模型打分（同 baseline predict.py）"""
    latest_date = data['日期'].max()
    sequences = []
    valid_stocks = []

    for sid in stock_ids:
        hist = data[(data['股票代码'] == sid) & (data['日期'] <= latest_date)]
        hist = hist.sort_values('日期').tail(config['sequence_length'])
        if len(hist) == config['sequence_length']:
            sequences.append(hist[features].values.astype(np.float32))
            valid_stocks.append(sid)

    if len(sequences) == 0:
        return {}, {}

    x = torch.from_numpy(np.array(sequences)).unsqueeze(0).to(device)
    with torch.no_grad():
        scores = model(x).squeeze(0).cpu().numpy()

    return valid_stocks, scores


def get_gbdt_scores(data, features, model, stock_ids):
    """GBDT 模型打分"""
    latest_date = data['日期'].max()
    X, valid_stocks = [], []

    for sid in stock_ids:
        stock_row = data[(data['股票代码'] == sid) & (data['日期'] == latest_date)]
        if len(stock_row) == 0:
            continue
        X.append(stock_row[features].values[0])
        valid_stocks.append(sid)

    if len(X) == 0:
        return [], []

    X = np.array(X, dtype=np.float32)
    scores = model.predict(X).flatten()
    return valid_stocks, scores


def main():
    data_file = os.path.join(config['data_path'], 'train.csv')
    model_dir = config['output_dir']
    transformer_path = os.path.join(model_dir, 'best_model.pth')
    scaler_trans_path = os.path.join(model_dir, 'scaler.pkl')
    scaler_gbdt_path = os.path.join(model_dir, 'scaler_gbdt.pkl')
    lgb_path = os.path.join(model_dir, 'lgb_model.txt')
    cat_path = os.path.join(model_dir, 'cat_model.cbm')
    output_path = os.path.join('./output/', 'result.csv')

    # 加载数据
    raw = pd.read_csv(data_file, dtype={'股票代码': str})
    raw['股票代码'] = raw['股票代码'].astype(str).str.zfill(6)
    raw['日期'] = pd.to_datetime(raw['日期'])
    latest_date = raw['日期'].max()
    stock_ids = sorted(raw['股票代码'].unique())
    stockid2idx = {sid: idx for idx, sid in enumerate(stock_ids)}

    feature_engineer = feature_engineer_func_map[config['feature_num']]
    # Transformer 训练时用了含 instrument 的197维特征
    trans_feat_cols = feature_cloums_map[config['feature_num']]  # 含 instrument
    # GBDT 模型使用不含 instrument 的196维特征
    gbdt_feat_cols = [c for c in trans_feat_cols if c != 'instrument']

    # 特征工程
    print("特征工程...")
    groups = [g for _, g in raw.groupby('股票代码', sort=False)]
    processed_list = []
    for g in groups:
        processed_list.append(feature_engineer(g.copy()))
    data = pd.concat(processed_list).reset_index(drop=True)
    data['instrument'] = data['股票代码'].map(stockid2idx)
    data = data.dropna(subset=['instrument']).copy()
    data['instrument'] = data['instrument'].astype(np.int64)

    # 填充缺失值
    data[trans_feat_cols] = data[trans_feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # GPU 设备
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    # ── 模型1: Transformer ──
    print("\n=== Transformer 预测 ===")
    scaler_t = joblib.load(scaler_trans_path)
    data_t = data.copy()
    data_t[trans_feat_cols] = scaler_t.transform(data_t[trans_feat_cols])
    model_t = StockTransformer(input_dim=len(trans_feat_cols), config=config, num_stocks=len(stock_ids))
    model_t.load_state_dict(torch.load(transformer_path, map_location=device, weights_only=True))
    model_t.to(device).eval()
    t_stocks, t_scores = get_transformer_scores(data_t, trans_feat_cols, model_t, scaler_t, stock_ids, device)

    # ── 模型2: LightGBM ──
    print("\n=== LightGBM 预测 ===")
    scaler_g = joblib.load(scaler_gbdt_path)
    data_g = data.copy()
    data_g[gbdt_feat_cols] = scaler_g.transform(data_g[gbdt_feat_cols])
    import lightgbm as lgbm
    lgb = lgbm.Booster(model_file=lgb_path)
    l_stocks, l_scores = get_gbdt_scores(data_g, gbdt_feat_cols, lgb, stock_ids)

    # ── 模型3: CatBoost ──
    print("\n=== CatBoost 预测 ===")
    from catboost import CatBoost
    cat = CatBoost()
    cat.load_model(cat_path)
    c_stocks, c_scores = get_gbdt_scores(data_g, gbdt_feat_cols, cat, stock_ids)

    # ── 集成融合 ──
    # 权重参考 Optiver 冠军: 0.5 CatBoost + 0.3 GRU + 0.2 Transformer
    # 根据验证集 final_score 调整: Transformer(0.069) > CatBoost(0.010) > LGB(-0.006)
    # 保守权重: Trans 0.6, Cat 0.25, LGB 0.15
    w_trans, w_cat, w_lgb = 0.5, 0.25, 0.25

    # 构建分数 dict
    score_dict = {}
    for i, s in enumerate(t_stocks):
        score_dict[s] = {'trans': float(t_scores[i])}
    for i, s in enumerate(l_stocks):
        score_dict.setdefault(s, {})['lgb'] = float(l_scores[i])
    for i, s in enumerate(c_stocks):
        score_dict.setdefault(s, {})['cat'] = float(c_scores[i])

    # 归一化各模型分数到 [0,1] 区间后加权
    def norm_scores(d, key):
        vals = [v.get(key, np.nan) for v in d.values() if key in v]
        if len(vals) < 5:
            return
        arr = np.array(vals)
        min_v, max_v = arr.min(), arr.max()
        if max_v - min_v < 1e-12:
            return
        for v in d.values():
            if key in v:
                v[key + '_norm'] = (v[key] - min_v) / (max_v - min_v)

    norm_scores(score_dict, 'trans')
    norm_scores(score_dict, 'lgb')
    norm_scores(score_dict, 'cat')

    # 加权平均
    ensemble = {}
    for sid, v in score_dict.items():
        s = 0.0
        w = 0.0
        if 'trans_norm' in v:
            s += w_trans * v['trans_norm']
            w += w_trans
        if 'cat_norm' in v:
            s += w_cat * v['cat_norm']
            w += w_cat
        if 'lgb_norm' in v:
            s += w_lgb * v['lgb_norm']
            w += w_lgb
        if w > 0:
            ensemble[sid] = s / w

    # 排序取 Top 5
    ranked = sorted(ensemble.items(), key=lambda x: x[1], reverse=True)
    top5 = ranked[:5]

    # ── 权重分配策略 ──
    # 策略1: proportional — 分数比例分配
    # 策略2: sqrt_prop — sqrt(分数) 比例（更平滑）
    # 策略3: rank_linear — 按排名线性递减 [5,4,3,2,1] 归一化
    strategy = config.get('weight_strategy', 'rank_linear')  # 验证集最优: 10.75%

    scores_arr = np.array([s for _, s in top5])
    if strategy == 'proportional':
        raw_weights = np.maximum(scores_arr, 0)  # 非负
        weights = raw_weights / (raw_weights.sum() + 1e-12)
    elif strategy == 'sqrt_prop':
        raw_weights = np.sqrt(np.maximum(scores_arr, 0))
        weights = raw_weights / (raw_weights.sum() + 1e-12)
    elif strategy == 'rank_linear':
        raw_weights = np.array([5, 4, 3, 2, 1], dtype=float)
        weights = raw_weights / raw_weights.sum()
    else:
        weights = np.ones(5) * 0.2

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result = pd.DataFrame({
        'stock_id': [s for s, _ in top5],
        'weight': [round(float(w), 4) for w in weights]
    })
    result.to_csv(output_path, index=False)

    print(f"\n预测日期: {latest_date.date()}")
    print(f"Top 5 股票: {[s for s, _ in top5]}")
    print(f"集成分数: {[round(sc, 4) for _, sc in top5]}")
    print(f"结果保存至: {output_path}")


if __name__ == '__main__':
    import multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    main()
