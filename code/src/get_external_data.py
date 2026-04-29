"""
外部数据集获取脚本
数据源: akshare (免费公开)
1. 申万行业分类 (SW industry classification)
2. 沪深300 PE/PB (index valuation)
3. 北向资金 (north-bound capital flow)

全部数据公开免费，符合竞赛要求。
"""
import os, sys, warnings
import pandas as pd
import numpy as np
import akshare as ak

warnings.filterwarnings('ignore')
DATA_DIR = os.path.join(os.path.dirname(__file__), '../../data/external')
os.makedirs(DATA_DIR, exist_ok=True)


def fetch_industry_classification():
    """
    获取申万行业分类历史（含 start_date，无 look-ahead bias）。
    输出: stock_code, industry_l1, industry_l2, effective_date
    """
    print("[1/3] 获取申万行业分类...")
    try:
        df = ak.stock_industry_clf_hist_sw()
        # df columns: symbol, start_date, industry_code, update_time
        df = df.rename(columns={'symbol': '股票代码', 'start_date': '生效日期'})
        df['股票代码'] = df['股票代码'].astype(str).str[-6:]  # extract 6-digit code
        df.to_csv(os.path.join(DATA_DIR, 'sw_industry.csv'), index=False)
        print(f"  行业分类: {len(df)} 条记录, {df['股票代码'].nunique()} 只股票")
        return df
    except Exception as e:
        print(f"  行业分类获取失败: {e}")
        return None


def fetch_index_valuation():
    """
    获取沪深300指数 PE/PB 历史（每日，可用于特征）。
    来源: 乐股乐股 (legulegu)
    """
    print("[2/3] 获取沪深300 PE/PB...")
    try:
        df = ak.stock_index_pe_lg(symbol="沪深300")
        df = df.rename(columns={
            '日期': '日期',
            '等权静态市盈率': 'pe_equal',
            '静态市盈率': 'pe_weighted',
            '滚动市盈率': 'pe_ttm',
            '等权市净率': 'pb_equal',
        })
        # 筛选需要的列
        cols = ['日期', 'pe_ttm', 'pe_weighted', 'pb_equal']
        df = df[[c for c in cols if c in df.columns]].copy()
        df.to_csv(os.path.join(DATA_DIR, 'hs300_valuation.csv'), index=False)
        print(f"  指数估值: {len(df)} 条记录, 日期 {df['日期'].min()} ~ {df['日期'].max()}")
        return df
    except Exception as e:
        print(f"  指数估值获取失败: {e}")
        return None


def fetch_northbound_flow():
    """
    获取北向资金历史净流入数据。
    来源: 东方财富
    """
    print("[3/3] 获取北向资金数据...")
    try:
        df = ak.stock_hsgt_hist_em(symbol="北向资金")
        # df columns: 日期, 当日成交净买额, 买入成交额, 卖出成交额, 累计净买额, ...
        df = df.rename(columns={
            '日期': '日期',
            '当日成交净买额': 'north_net_buy',
            '买入成交额': 'north_buy',
            '卖出成交额': 'north_sell',
            '累计净买额': 'north_cum_buy',
        })
        cols = ['日期', 'north_net_buy', 'north_cum_buy']
        df = df[[c for c in cols if c in df.columns]].copy()
        df['日期'] = pd.to_datetime(df['日期'])
        df.to_csv(os.path.join(DATA_DIR, 'northbound.csv'), index=False)
        print(f"  北向资金: {len(df)} 条记录, 日期 {df['日期'].min().date()} ~ {df['日期'].max().date()}")
        return df
    except Exception as e:
        print(f"  北向资金获取失败: {e}")
        return None


def main():
    print("=" * 50)
    print("外部数据集获取")
    print(f"输出目录: {DATA_DIR}")
    print("=" * 50)

    industry = fetch_industry_classification()
    valuation = fetch_index_valuation()
    north = fetch_northbound_flow()

    print("\n完成！文件列表:")
    for f in sorted(os.listdir(DATA_DIR)):
        size = os.path.getsize(os.path.join(DATA_DIR, f))
        print(f"  {f}: {size/1024:.1f}KB")


if __name__ == '__main__':
    main()
