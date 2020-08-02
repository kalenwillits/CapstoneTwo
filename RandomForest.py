import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score
cd_data = 'data/'

X_train = pd.read_csv(cd_data+'X_train_oh.csv')
X_test = pd.read_csv(cd_data+'X_test_oh.csv')
y_train = pd.read_csv(cd_data+'y_train_oh.csv')
y_test = pd.read_csv(cd_data+'y_test_oh.csv')

model = RandomForestClassifier()

def binary_delays(data, threshold=0.15):
    data = data.replace(data[data['DelayRatio'] >= threshold], 1)
    data = data.replace(data[data['DelayRatio'] <= threshold], 0)
    return data

y_train_bi = binary_delays(y_train)
y_test_bi = binary_delays(y_test)

model.fit(X_train, y_train_bi)
y_pred = model.predict(X_test)
plt.figure(figsize=(10,10))
plt.scatter(X_test['FlightDate'], y_test, c=y_pred)
plt.title('RandomForestClassifier delays < 15%')
plt.xlabel('Day of Flight Date in Januwary 2017')
plt.ylabel('Delay Ratio')
plt.legend()
plt.show()
