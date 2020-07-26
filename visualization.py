import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

cd_data = 'data/'
file = 'flight_delays.csv'
df = pd.read_csv(cd_data+file)
df.head()

print('\n___________Statistics___________\n', df.describe())

df_numbers_only = df.drop('UniqueCarrier', axis=1)
sns.heatmap(df_numbers_only.corr())
plt.savefig('figures/Heatmap.png')
