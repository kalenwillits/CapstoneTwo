import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import Lasso

cd_data = 'data/'

X_train = pd.read_csv(cd_data+'X_train_Pacific.csv')
X_test = pd.read_csv(cd_data+'X_test_Pacific.csv')
y_train = pd.read_csv(cd_data+'y_train_Pacific.csv')
y_test = pd.read_csv(cd_data+'y_test_Pacific.csv')

model = Lasso(alpha=0.2)

model.fit(X_train[['FlightDate']], y_train)
y_pred = model.predict(X_test[['FlightDate']])
plt.scatter(X_test[['FlightDate']], y_test)
plt.plot(X_test[['FlightDate']], y_pred, color='red')
