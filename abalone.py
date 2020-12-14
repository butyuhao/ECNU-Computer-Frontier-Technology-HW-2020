import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import datasets, linear_model
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# 读取训练集和测试集
train_data = pd.read_csv('./data/train.csv', encoding='utf-8', header=None)
test_data = pd.read_csv('./data/test.csv', encoding='utf-8', header=None)

# 将字母类型的数据替换成数字，M,F,I分别替换为1，2，3
train_data = train_data.replace(to_replace=['M', 'F', 'I'], value=[1, 2, 3])
test_data = test_data.replace(to_replace=['M', 'F', 'I'], value=[1, 2, 3])

# 从训练数据中提取X和y，X是0-7列，我们要预测的是鲍鱼的年龄，在第8列
X = train_data.iloc[:, 0:8]
y = train_data.iloc[:, 8]

X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=1, test_size=0.1)

linreg = LinearRegression()
linreg.fit(X_train, y_train)

print("最佳拟合线的截距："+str(linreg.intercept_))
print("回归系数："+str(linreg.coef_))

y_pred = linreg.predict(X_val)
print ("MSE:",metrics.mean_squared_error(y_val, y_pred))
print ("RMSE:",np.sqrt(metrics.mean_squared_error(y_val, y_pred)))

y_pred_test = linreg.predict(test_data).tolist()

y_pred_test = [round(p, 2) for p in y_pred_test]

with open('result.txt', 'w') as f:
    for r in y_pred_test:
        f.write(str(r) + '\n')
