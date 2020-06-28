# %% codecell
import pandas as pd
import numpy as np
import pandas_profiling
# %% codecell
!pwd
!ls -a
!ls data -a
# %% codecell
!pwd
path_to_data = 'data/raw/On_Time_On_Time_Performance_2017_1.csv'
df = pd.read_csv(path_to_data, low_memory=False)
# %% codecell
df.columns
# %% codecell
#df.plot()
# %% codecell
df.info()
# %% codecell
df.describe().transpose()
# %% codecell
df.isnull().sum()
# %% codecell
