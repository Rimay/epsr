import pandas as pd

# t = pd.Series([1,2,3,4,5])

# # print(t.rolling(window=3).mean())
# # print(t.rolling(window=3).std())
# print(t.shift(1))
# print(t[0:2])

# 创建一个按天记录的时间序列
idx = pd.date_range('2026-11-01', periods=5, freq='D')
s = pd.Series([10, 20, 30, 40, 50], index=idx)
print("原始数据:\n", s)

# 转换为按工作日 (Business day, 'B')
s_b = s.asfreq('B')
print("\n转换为工作日频率:\n", s_b)
