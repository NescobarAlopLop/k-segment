from mat4py import loadmat
import pandas as pd
import os
print(os.path.realpath(__file__))

name_withou_ext = '/home/ge/k-segment/datasets/anomaly/satellite'
data = loadmat(name_withou_ext + '.mat')

print(type(data))
data = pd.DataFrame.from_dict(data)
a = ['c'+ str(x) for x in range(len(data.iloc[0,0]))]
data[a] = pd.DataFrame(data.X.values.tolist(), data.index)
data[['Label']] = pd.DataFrame(data.y.values.tolist(), index= data.index)
data.drop(['X', 'y'], axis=1, inplace=True)
data['Label'] = data['Label'].astype(int)
data['Label'] += 1
data.to_csv(name_withou_ext + ".csv", index=False, encoding='utf8')
print(type(data))

print(data.head(5))
print(data.dtypes)
print(data.describe())
