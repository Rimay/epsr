import pandas as pd

t = pd.Series([1,2,3,4,5])

# print(t.rolling(window=3).mean())
# print(t.rolling(window=3).std())
print(t.shift(1))
print(t[0:2])