import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

file = open('output.txt', 'w+')
df = pd.read_csv('data/On_Time_On_Time_Performance_2017_1.csv', low_memory=False)
cols = df.columns.tolist()
print(cols)

df.plot(kind='scatter', x='DepDelay', y='ArrDelay')
plt.show()
