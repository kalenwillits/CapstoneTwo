# %% codecell
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

# %% codecell
path = 'data/interim/'
file = 'flight_delays.csv'
df = pd.read_csv(path+file)
df.head()

# %% codecell
print('Shape:\n', df.shape)
print('Info:\n', df.info())
print('Statistics:\n', df.describe())

# %% codecell
df_numbers_only = df.drop('UniqueCarrier', axis=1)
sns.heatmap(df_numbers_only.corr())
plt.savefig('figures/Heatmap.png')

# %% markdown

### Heatmap Observations
# - Wac shows a correlation, however this is an ID according to Column_info.txt and can be igonored.
# - There is a correlation between departure time and arrival time that is nearly common sense, however further investigation is certainly warrented.
# - The distance and air time columns have a strong correlation. This is not all that surprising, however this can be investigated with other parameters.
# - Departure delay and late aircraft delay is correlated. Likely an easy prediciton once the aircraft delay is accounced for but we are looking to predict the initial delay.
# - ~~A diverted aircraft lines up with late aicraft delays, arrival delays, carrier delays, and NAS delays. According to the data set's information the NAS is "National Air System Delay, in Minutes".~~
#    - After resampling late aircraft delays and airtime, these correlations did not appear in the heat map.
# - The departure and late aircraft correlation is suprisingly weak.
# What we are really interested in here is the delays and other causes that can contribute to these recorded delays.


# %% codecell
plt.figure(figsize=(10,10))
plt.scatter(df['DepTime'], df['ArrTime'], alpha=0.1, color='blue', s=0.03)
plt.title('Departure Time VS Arrival Time')
plt.xlabel('Departure Time')
plt.ylabel('Arrival Time')
plt.savefig('figures/DepTime_VS_ArrTime.png')
# %% markdown
# As expected there is a very strong correlation with departure time and arrival time. This could be clearly predicted with a linear regression model and certainly could be a piece in later analysis.
# %% codecell
plt.figure(figsize=(10,10))
plt.scatter(df['Distance'], df['AirTime'], alpha=0.1, color='blue', s=0.8)
plt.title('Distance VS Air Time')
plt.xlabel('Distance (Miles)')
plt.ylabel('Air Time (Minutes)')
plt.savefig('figures/Distance_VS_AirTime.png')
# %% markdown
# The linear regression is even more pronounced here showing a narrow range of posibilities. There seems to also be a straight line hovering above 100 minutes of airtime with a range of distance.
# This could mean that there is an issue in the reported data. Likely where I filled Nan values with Means.
# %% codecell
plt.figure(figsize=(10,10))
plt.scatter(df['DepDelay'], df['LateAircraftDelay'], alpha=0.5, color='blue', s=2)
plt.title('Departure Delay VS Late Aircraft')
plt.xlabel('Departure Delay (Minutes)')
plt.ylabel('Late Aircraft (Minutes)')
plt.savefig('figures/DepDelay_VS_LateAircraft.png')
# %% markdown
# Again there is a sharp linear regression with most data points sitting below 400 minutes.
# Several outliers sit well beyond that mark but are still within the regression.
# Many data points are showing a late aircraft delay of 0. Which is remarkable even after resampling missing data.
# This suggests that departure delays are not always a causation of a late aircraft.
# %% codecell
df.columns
plt.figure(figsize=(10,10))
df['Diverted'].value_counts()
# %% markdown
# It appears that the diverted correlations are not interesting because the diverted column is a boolean value.
# This may be more useful later on, but for now it does not help.

# %% markdown
# ## Scatterplot Observations
# Now that I have had some observations on the measurements on flight delays, I can start attacking the date time objects and looking for when one flight delay will cause another.
# My hypothosis is: " If a flight is delayed from reaching it's destination, the sequential flight leaving it's destination will likely be delayed."
# In order to test my hypthosis, I will need to narrow down on one airport and find some flight delays. Then I can follow those flights and look for these chain reaction delays causing other flights to be delayed.
