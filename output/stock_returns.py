"""详细计算每只股票的收益率"""
import pandas as pd
import os
import json

os.chdir(r'E:\stock\baseline_repo')

# 读取数据
result = pd.read_csv('output/result.csv')
test = pd.read_csv('data/test.csv')

# 转换股票代码格式 - 将"002422"转为2422
result['stock_code_int'] = result['stock_id'].astype(str).apply(lambda x: int(x) if x.isdigit() else int(x.lstrip('0')))

# 计算每只股票的收益率
def calculate_return(group):
    start = group.iloc[0]
    end = group.iloc[-1]
    return (end['开盘'] - start['开盘']) / start['开盘']

# 获取最后5条记录并计算收益率
test_last5 = test.groupby('股票代码').tail(5)

# 对每只预测股票计算收益率
returns = {}
for idx, row in result.iterrows():
    stock_code = row['stock_code_int']
    stock_data = test_last5[test_last5['股票代码'] == stock_code]
    if len(stock_data) > 0:
        ret = calculate_return(stock_data)
        returns[row['stock_id']] = {
            'weight': row['weight'],
            'return': ret,
            'weighted_return': ret * row['weight']
        }

print("=" * 70)
print("每只预测股票的收益率分析")
print("=" * 70)
print(f"{'股票代码':<12} {'权重':<10} {'收益率':<15} {'加权收益':<15}")
print("-" * 70)
total_weighted = 0
for stock_id, data in returns.items():
    print(f"{stock_id:<12} {data['weight']:<10.4f} {data['return']:<15.4%} {data['weighted_return']:<15.6%}")
    total_weighted += data['weighted_return']

print("-" * 70)
print(f"{'总计':<12} {sum(d['weight'] for d in returns.values()):<10.4f} {'':<15} {total_weighted:<15.4%}")
print()

# 读取基线参数
with open('model/60_158+39/best_params.json', 'r') as f:
    baseline_params = json.load(f)

print("=" * 70)
print("模型配置信息")
print("=" * 70)
print(f"最佳参数: {json.dumps(baseline_params['params'], indent=2)}")
print(f"验证集分数: {baseline_params['value']:.6f}")
print()

# 读取最终分数
with open('model/60_158+39/final_score.txt', 'r') as f:
    print("Final Score文件内容:")
    print(f.read())

print()
print("=" * 70)
print("基线模型对比")
print("=" * 70)
print("当前模型使用CatBoost单模型 (use_cat=true, w_cat=0.5487)")
print(f"验证集 Final Score: 0.069245")
print(f"测试集实际加权收益率: {total_weighted:.4%}")
print()
print("基线模型对比说明:")
print("1. 项目中默认权重策略为'equal'（等权重0.2）")
print("2. 当前模型优化后的最佳参数使用加权策略")
print("3. 验证集得分0.069245表示模型在验证期间的表现")
print("4. 测试集实际收益1.4%为真实预测收益")