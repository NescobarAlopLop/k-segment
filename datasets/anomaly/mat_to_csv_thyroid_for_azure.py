from mat4py import loadmat
import pandas as pd


data = loadmat('thyroid.mat')

print(type(data))
data = pd.DataFrame.from_dict(data)
a = ['c'+ str(x) for x in range(len(data.iloc[0,0]))]
data[a] = pd.DataFrame(data.X.values.tolist(), data.index)
data[['Label']] = pd.DataFrame(data.y.values.tolist(), index= data.index)
data.drop(['X', 'y'], axis=1, inplace=True)
data['Label'] = data['Label'].astype(int)
data['Label'] += 1
data.to_csv("thyroid.csv", index=False, encoding='utf8')
print(type(data))

print(data.head(5))
print(data.dtypes)
print(data.describe())