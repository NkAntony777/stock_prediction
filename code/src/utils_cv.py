"""
截面特征增强模块 — 横截面中性化、行业变量、市场指标
参考 Kaggle Optiver 第7名"Deviation 特征"思路
"""
import pandas as pd
import numpy as np


def add_cross_sectional_features(df, group_col='日期'):
    """
    对已有特征做截面中性化，生成 Deviation 特征。
    对每个日期的所有股票，计算每个特征值与该日中位数的偏差。

    这是 Optiver 第7名方案的核心操作：原始特征 - 截面中位数 → 保留个股特异信号
    """
    df = df.copy()
    # 找出所有数值特征列
    exclude = {'instrument', '股票代码', '日期', 'label', 'open_t1', 'open_t5',
               'stock_idx', 'datetime'}
    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                if c not in exclude]

    # 只对模型特征做偏差（避免对 label 等做）
    model_cols = [c for c in num_cols if c not in ('开盘', '收盘', '最高', '最低',
                  '成交量', '成交额', '振幅', '涨跌额', '换手率', '涨跌幅')]

    new_dfs = []
    for date, group in df.groupby(group_col, sort=False):
        group = group.copy()
        for col in model_cols:
            median_val = group[col].median()
            if pd.notna(median_val) and median_val != 0:
                # Deviation: 原始值 - 截面中位数
                group[f'{col}_dev'] = group[col] - median_val
                # Ratio: 原始值 / 截面中位数 (带符号)
                group[f'{col}_rank'] = group[col].rank(pct=True)
        new_dfs.append(group)

    result = pd.concat(new_dfs).reset_index(drop=True)
    result.fillna(0, inplace=True)
    return result


def add_market_features(df, group_col='日期'):
    """
    添加市场整体特征（每个交易日的市场统计量）。
    帮助模型感知整体市场环境。
    """
    df = df.copy()

    # 市场平均收益率
    df['market_return'] = df.groupby(group_col)['涨跌幅'].transform('mean')
    df['market_volatility'] = df.groupby(group_col)['涨跌幅'].transform('std')

    # 上涨家数占比
    df['up_ratio'] = df.groupby(group_col)['涨跌幅'].transform(
        lambda x: (x > 0).mean()
    )

    # 成交量放大程度（市场量比）
    if '成交量' in df.columns:
        df['market_volume'] = df.groupby(group_col)['成交量'].transform('sum')

        # 滚动5日市场量比
        market_daily = df.groupby(group_col)['成交量'].sum().reset_index()
        market_daily = market_daily.sort_values(group_col)
        market_daily['market_vol_ma5'] = market_daily['成交量'].rolling(5).mean()
        market_daily['market_vol_ratio'] = market_daily['成交量'] / (market_daily['market_vol_ma5'] + 1)

        vol_map = dict(zip(market_daily[group_col], market_daily['market_vol_ratio']))
        df['market_vol_ratio'] = df[group_col].map(vol_map).fillna(1.0)

    # 宽度指标：涨跌幅分布
    df['return_dispersion'] = df.groupby(group_col)['涨跌幅'].transform(
        lambda x: np.percentile(x, 80) - np.percentile(x, 20)
    )

    df.fillna(0, inplace=True)
    return df


def get_enhanced_feature_list(base_features):
    """获取增强后完整特征列表（用于模型训练时取子集）"""
    # 基础特征
    feats = list(base_features)
    # 新增的截面特征后缀
    suffixes = ['_dev', '_rank']

    model_cols = [c for c in base_features if c not in ('开盘', '收盘', '最高', '最低',
                   '成交量', '成交额', '振幅', '涨跌额', '换手率', '涨跌幅')]
    for col in model_cols:
        for suffix in suffixes:
            feats.append(f'{col}{suffix}')

    # 市场特征
    feats.extend(['market_return', 'market_volatility', 'up_ratio',
                  'market_vol_ratio', 'return_dispersion'])

    return feats


def engineer_features_enhanced(df):
    """
    替代原有的 engineer_features_158plus39，增加截面特征。
    用于在训练/预测前调用。
    """
    from utils import engineer_features_158plus39

    # 先计算基线特征
    df = engineer_features_158plus39(df.copy())

    # 添加截面特征
    df = add_market_features(df)
    df = add_cross_sectional_features(df)

    return df
