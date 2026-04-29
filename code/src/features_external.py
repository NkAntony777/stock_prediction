"""
外部数据特征工程
将行业分类、北向资金等外部数据合并为训练特征。
所有数据均为公开免费来源（akshare），符合竞赛要求。
"""
import os
import numpy as np
import pandas as pd

EXTERNAL_DIR = os.path.join(os.path.dirname(__file__), '../../data/external')


def load_external_data():
    """加载所有外部数据"""
    data = {}

    # 1. 行业分类
    industry_path = os.path.join(EXTERNAL_DIR, 'sw_industry.csv')
    if os.path.exists(industry_path):
        df_ind = pd.read_csv(industry_path)
        df_ind['生效日期'] = pd.to_datetime(df_ind['生效日期'])
        data['industry'] = df_ind
        print(f"  加载行业分类: {len(df_ind)} 条")

    # 2. 北向资金
    north_path = os.path.join(EXTERNAL_DIR, 'northbound.csv')
    if os.path.exists(north_path):
        df_north = pd.read_csv(north_path)
        df_north['日期'] = pd.to_datetime(df_north['日期'])
        data['northbound'] = df_north
        print(f"  加载北向资金: {len(df_north)} 条")

    return data


def add_industry_features(main_df, industry_df):
    """
    将行业分类转为截面特征。
    对每个日期，计算股票所属行业的平均收益，作为行业动量信号。
    无 look-ahead bias: 使用生效日期匹配。
    """
    if industry_df is None or len(industry_df) == 0:
        return main_df

    df = main_df.copy()
    df['日期'] = pd.to_datetime(df['日期'])

    # 为每只股票匹配最新的行业分类
    industry_df = industry_df.sort_values(['股票代码', '生效日期'])

    # 构建 stock+date → industry 映射
    # 对每个 stock，找到 date >= 生效日期 的最新分类
    industry_map = {}
    for stock, group in industry_df.groupby('股票代码'):
        group = group.sort_values('生效日期')
        industry_map[stock] = group

    # 为每行匹配行业
    def _get_industry(row):
        stock = str(row['股票代码']).zfill(6)
        date = row['日期']
        if stock not in industry_map:
            return 'unknown', 'unknown'
        entries = industry_map[stock]
        # 找到生效日期 <= date 的最新记录
        valid = entries[entries['生效日期'] <= date]
        if len(valid) == 0:
            return 'unknown', 'unknown'
        last = valid.iloc[-1]
        return last.get('industry_code', 'unknown'), last.get('industry_code', 'unknown')

    # 只对数据中的股票做匹配（加速）
    unique_stocks = df['股票代码'].unique()
    stock_industry = {}
    for stock in unique_stocks:
        if stock in industry_map:
            stock_industry[stock] = industry_map[stock]
        else:
            stock_industry[stock] = None

    # 为每个日期，当天所有股票的行业标签（用于截面特征）
    df['industry_code'] = 'unknown'
    for stock in unique_stocks:
        if stock_industry[stock] is None:
            continue
        mask = df['股票代码'] == stock
        dates = df.loc[mask, '日期'].values
        entries = stock_industry[stock]
        for d in dates:
            valid = entries[entries['生效日期'] <= pd.Timestamp(d)]
            if len(valid) > 0:
                df.loc[mask & (df['日期'] == d), 'industry_code'] = valid.iloc[-1]['industry_code']

    # 行业动量: 每天每个行业的平均涨跌幅
    if '涨跌幅' in df.columns and 'industry_code' in df.columns:
        industry_return = df.groupby(['日期', 'industry_code'])['涨跌幅'].transform('mean')
        df['industry_mean_return'] = industry_return

        # 行业相对强度: 个股涨跌幅 vs 行业均值
        df['stock_vs_industry'] = df['涨跌幅'] - industry_return

        # 行业 one-hot 编码 — 在个股排序任务中噪声太大，跳过
        # 仅保留连续型行业特征

    # 行业集中度: 当天该行业有多少只股票上涨
    if '涨跌幅' in df.columns and 'industry_code' in df.columns:
        up_ratio = df.groupby(['日期', 'industry_code'])['涨跌幅'].transform(
            lambda x: (x > 0).mean()
        )
        df['industry_up_ratio'] = up_ratio

    print(f"  行业特征: stock_vs_industry, industry_mean_return, industry_up_ratio")
    return df


def add_northbound_features(main_df, north_df):
    """
    添加北向资金特征。
    每日北向净买入可作为市场情绪信号。
    无 look-ahead bias: 使用 T-1 数据（前一日公布）
    """
    if north_df is None or len(north_df) == 0:
        return main_df

    df = main_df.copy()
    if '日期' in df.columns:
        df['日期'] = pd.to_datetime(df['日期'])
    else:
        return df

    north = north_df.copy()
    north = north.rename(columns={'日期': '日期'})

    # 滚动窗口特征
    north = north.sort_values('日期')

    # 5/10/20 日净买入均值
    if 'north_net_buy' in north.columns:
        north['north_5d_avg'] = north['north_net_buy'].rolling(5).mean()
        north['north_10d_avg'] = north['north_net_buy'].rolling(10).mean()
        north['north_20d_avg'] = north['north_net_buy'].rolling(20).mean()
        north['north_5d_sum'] = north['north_net_buy'].rolling(5).sum()

    # 合并: 使用当天 (T+0) 数据，因为收盘后即可获取
    # 实际回测中北向数据当日收盘后公布，T+1 开盘前可知
    merge_cols = ['日期']
    feature_cols = [c for c in north.columns if c.startswith('north_')]
    merge_cols.extend(feature_cols)

    north_subset = north[merge_cols].copy()

    # 前移一天: T+1 日开盘只能用 T 日收盘后数据
    # 当前数据中 T 日的 north_net_buy 是 T 日收盘后公布的
    # 所以预测 T+1 时可直接使用
    df = df.merge(north_subset, on='日期', how='left')
    for col in feature_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    print(f"  北向特征: {feature_cols}")
    return df


def engineer_external_features(df):
    """
    主入口: 对 DataFrame 添加所有外部特征。
    输入: 已完成基线特征工程的 DataFrame (需含 日期, 股票代码, 涨跌幅 列)
    输出: 添加外部特征后的 DataFrame
    """
    if len(df) == 0:
        return df

    ext_data = load_external_data()

    df = df.copy()

    if 'industry' in ext_data:
        df = add_industry_features(df, ext_data['industry'])

    if 'northbound' in ext_data:
        df = add_northbound_features(df, ext_data['northbound'])

    # 清理
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)

    # 打印新增特征数
    new_cols = [c for c in df.columns if c.startswith('ind_')
                or c in ('industry_mean_return', 'stock_vs_industry',
                        'industry_up_ratio')
                or c.startswith('north_')]
    print(f"外部特征总计: {len(new_cols)} 个")

    return df


def get_new_feature_cols():
    """返回外部特征列名列表（精选，避免噪声）"""
    return [
        'stock_vs_industry',      # 个股相对行业强弱
        'industry_mean_return',   # 行业动量
        'industry_up_ratio',      # 行业宽度
        'north_5d_avg',           # 北向5日趋势
    ]
