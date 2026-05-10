"""计算股票收益率和基线对比的脚本"""
import pandas as pd
import os

# 确保在正确的工作目录
os.chdir(r'E:\stock\baseline_repo')

# 读取数据
result = pd.read_csv('output/result.csv')
test = pd.read_csv('data/test.csv')

print("=" * 60)
print("RESULT.CSV - 模型预测的股票")
print("=" * 60)
print(result.to_string())
print()

print("=" * 60)
print("TEST.CSV - 测试集股票代码样例")
print("=" * 60)
print("唯一股票数量:", test['股票代码'].nunique())
print("股票代码样例:", sorted(test['股票代码'].unique())[:15])
print()

# 提取预测的股票代码
predicted_stocks = result['stock_id'].astype(str).tolist()
print("预测股票代码(字符串):", predicted_stocks)

# 检查test.csv中的股票代码格式
test_codes = test['股票代码'].unique()
print("\n测试集股票代码样例(前15):", sorted(test_codes)[:15])

# 检查result.csv中的股票代码是否在test.csv中
# 首先转换result中的代码为整数
try:
    result_codes_int = [int(float(code)) for code in result['stock_id']]
    print("\n预测股票代码(整数):", result_codes_int)
    
    # 检查匹配
    matched = [c for c in result_codes_int if c in test['股票代码'].unique()]
    print(f"匹配的股票数量: {len(matched)}")
except Exception as e:
    print(f"转换错误: {e}")

print("\n" + "=" * 60)
print("SCORE_SELF.PY 执行结果")
print("=" * 60)
# 运行score_self.py
os.system('.venv\\Scripts\\python.exe test\\score_self.py')