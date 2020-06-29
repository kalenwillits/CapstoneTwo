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
# Again there is a sharp linear regression with most data points sitting below 200 minutes.
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
# Now that I have had some observations on the measurements on flight delays, I can start attacking the date time objects and looking for when one flight delay that will cause another.
# My hypothosis is: " If a flight is delayed from reaching it's destination, the sequential flight leaving it's destination will likely be delayed."
# Luckily the column "Late Aircraft Delay" is a measurment of exactly the kind of delay that I am looking for.
# In order to test my hypthosis, I will need to narrow down on one airport and find some flight delays. Then I can follow those flights and look for these chain reaction delays causing other flights to be delayed.
# %% codecell
df.columns
# A single delay paramter.
df_delays_sum = df['CarrierDelay'] + df['ArrDelay'] + df['NASDelay'] + df['DepDelay'] + df['SecurityDelay'] + df['LateAircraftDelay']

df['DelaySum'] = df_delays_sum
df['DelaySum'] = df['DelaySum'].fillna(0)
#Filling the Nan delays with zero as to not throw off the average.

flight_info = ['Origin', 'Dest', 'FlightDate', 'DelaySum']
print(df[df['DelaySum'] ==  df['DelaySum'].max()][flight_info])
print(df['DelaySum'].max()/60/24)
# %% markdown
# ### Where to look
# > The flight that experinced the longest delay in 2017 was from Baltimore/Washington International Airport and going to Dallas/Fort Worth International Airport.
# The Delay was just over 4 days and the actual flight date took place on January 11th.
# It just so happens that on January 7th there was a severe winter storm orginating 78 Nautical Miles South West of Baltimore at the Philadelphia International Airport that caused flight delays that left passengers without an estimate.
# The article on the storm in Philadelphia is sited here: https://philly.metro.us/weather-causes-delays-cancellations-at-philly-airport/
#
# I believe we can expect this to be an outlier. Under these extreme conditions, the recovery for the flight delay caused by this aircraft was doubtfully the entire 4 days. What I am looking for is to predict most of the flight delays, not just the extreme weather delays.
# So from here I will move on to grouping the data by desination and finding the maximum average delay by flight destination.
# %% codecell
df['DelaySum'].describe()
df_group = df.groupby('Dest')
df_group[flight_info].head()

destinations = list(df['Dest'].unique())
groups_mean = {}
for destination in destinations:
    groups_mean[destination] = [df[df['Dest'] == destination]['DelaySum'].mean()]
df_group = pd.DataFrame(groups_mean).max()
df_group[df_group == df_group.max()]
# ELM = Elmira Corning Regional Airport - New York
# DTW = Detroit Metropolitan Airport

df[df['Dest'] == 'ELM'][flight_info]
# Interesting. The higest average of delays contains flight from Elmira Corning Regional Airport to Detroit Metropolitan Airport.
# Lets check out DTW to see if my hypothosis holds true.
df[df['Dest'] == 'DTW']['DelaySum'].mean()
df[df['Dest'] == 'DTW'][flight_info]
df[(df['Dest'] == 'ELM') & (df['DelaySum'] > 0)]['DelaySum'].count()
# Number of flight delays for any reason for flights going to Detroit Metropolitan Airport

# %% markdown
# I supsect the reason that I found this airport is because it does not have many data points at all!
# The central limit theorem would suggest that I have found this smaller airport by chance. The one long flight delay is wildely skewing the data.
# Lets instead attempt to parse through the same groups of airports by flight delay counts greater than 0 instead.
# %% codecell


destinations = list(df['Origin'].unique())
groups_count = {}
for destination in destinations:
    groups_count[destination] = df[(df['Origin'] == destination) & (df['DelaySum'] > 0)].count()

df_group = pd.DataFrame(groups_count).max()
df_group[df_group == df_group.max()]
# %% markdown
# This shows that the Los Angelos International Airport was the biggest offender of incoming flight delays.
# This of course is a prosecuter's fallacy because LAX happens to also be one of the busiest airports.
# Either way it makes a good place to start.
#
# From here the experiement will work as follows:
# - Treat a flight delay like it has a half life and count the number of relational delays going out of LAX to the next similar airport.
# - Then we can expect that there is a chance this amount of delays will cause more delays at the next airport.
# This of course is not as narrow enough of a scope to complete the analysis, however if this is correct we can change our hypothosis to affect flights for relational flights.
# After running the above code again retuning it for origin, it shows that Atlanta International Airport is the biggest offender for delays in 2017.
# I'm going to start there, write a funtion that follows the highest amount of flight delays, plot that, and create a linear regression.
# %% codecell
# here I will code the first few steps manually.


# It's time to turn that previous code into a proper function.
def count_delays(data, type='Dest'):
    """ Takes the 2017 flight database and counts each flight delay going to a paticular group """
    #origins = list(df['Origin'].unique())
    destinations = list(data[type].unique())
    num_delays = {}
    for destination in destinations:
        num_delays[destination] = data[(data[type] == destination) & (data['DelaySum'] > 0)].count()
        return pd.DataFrame(num_delays)

atl_group = df[df['Origin'] == 'ATL'].groupby('Origin')
atl_group[flight_info].head()

atl_group = df[df['Origin'] == 'ATL']
I = count_delays(atl_group)
I_dest = I.columns[0]
print(I.values[0])
print(I_dest)

II = count_delays(df[df['Origin'] == 'PHX'])
II_dest = II.columns[0]
print(II.values[0])
print(II_dest)


origins = list(df['Origin'].unique())
destinations = list(df['Dest'].unique())

delays_origins = []
for origin in origins:
    delays_count = df[(df['Origin'] == origin) & (df['DelaySum'] > 0)].count()['DelaySum']
    delays_origins.append(delays_count)

len(delays_origins)

delays_destinations = []
for destination in destinations:
    delays_count = df[(df['Dest'] == destination) & (df['DelaySum'] > 0)].count()['DelaySum']
    delays_destinations.append(delays_count)

delays_destinations.append(np.mean(delays_destinations))


# delay decay
# In this case, coming from Atlanta to Phoenix Sky Harbor International Airport generated the most flight delays. Lets keep going.
# This works, now lets plot it.


# %% codecell
# Turn the DelaySum into a Boolean Value.
# Create ratios for all destinations num_of_delays/num_of_flights then all origins
# Then plot and hope it's a linear regression.
jan17_flights_lax = df[(df['FlightDate'] == '2017-01-17') & (df['Dest'] == 'LAX')]
lax_flight_date = df[df['Dest'] == 'LAX']['FlightDate']
lax_flight_delay = df[df['Dest'] == 'LAX']['DelaySum']
plt.scatter(lax_flight_date, lax_flight_delay)

delay_ratio = jan17_flights_lax[jan17_flights_lax['DelaySum'] > 0]['DelaySum'].count()/jan17_flights_lax['DelaySum'].count()
delay_ratio

# %% markdown
# After much exploring, I believe that I will need to creat a delay ratio data set from a single location. LAX has enough points to do this.
# %% codecell

def calculate_origin_ratios(origin, df=df, dates=dates, flight_info=flight_info):
    """
    df = The default DataFrame
    dates = taken as a list of datetime objects
    origin = specifies which Origin to choose in the origin column.
    flight_info = ['Origin', 'Dest', 'FlightDate', 'DelaySum']
        The purpose of this list was to make the dataframe smaller
        and easier to work with for specific operations
    _______________________________________________________________
    The purpose of this function is to take flight data and return a
    flight delay ratio by origin.
    """
    result = {
    'Origin':[],
    'FlightDate':[],
    'DelayRatio':[],
    }
    for date in dates:

        flights_by_date = df[(df['FlightDate'] == date) & (df['Origin'] == origin)][flight_info]
        num_delays = flights_by_date[flights_by_date['DelaySum'] > 0]['DelaySum'].count()
        num_flights = flights_by_date['DelaySum'].count()

        try:
            delay_ratio = num_delays/num_flights
        except ZeroDivisionError:
            # If there are no flights at an airport there is a possibility to divide by 0
            delay_ratio = 0

        result['Origin'].append(origin)
        result['FlightDate'].append(date)
        result['DelayRatio'].append(delay_ratio)
    return(pd.DataFrame(result))
# %% codecell
dates = list(df['FlightDate'].unique())
#TOP 6 airports Seattle is actually 7th on the list, however I wanted to even out my plot geographically.
origins = ['ATL', 'LAX', 'ORD', 'DFW', 'DEN', 'SEA']
plt.figure(figsize=(20,20))
plt.grid()
for origin in origins:
    origin_ratio = calculate_origin_ratios(origin)
    plt.plot(origin_ratio['FlightDate'], origin_ratio['DelayRatio'], label=origin)
plt.xlabel('Flight Date')
plt.xticks(rotation=90)
plt.ylabel('Delay Ratio')
plt.title('Origin Delay Ratios by Date')
plt.legend()
plt.savefig('figures/DelayRatios.png')
# %% markdown
# ### Application
# Now that there is a linear figure to show just how many delays happen where and when, I can test my theory that delays at one origin will affect delays at the next destination.
# If we go back to the flight info filter and check LAX to ATL, we can reference the plot and attempt to find delays from ATL to other airports that may have orginated with the LAX delay.
# %% codecell
lax_atl = df[(df['Origin'] == 'LAX') | (df['Origin'] == 'ATL')][flight_info]
lax_atl['Dest'].value_counts()
# SFO appears to have the most destinations of flights from LAX and ATL. We should expect the delays to correlate.
dates = list(df['FlightDate'].unique())
#TOP 6 airports Seattle is actually 7th on the list, however I wanted to even out my plot geographically.
origins = ['ATL', 'LAX', 'SFO']
plt.figure(figsize=(20,20))
plt.grid()
for origin in origins:
    origin_ratio = calculate_origin_ratios(origin)
    plt.plot(origin_ratio['FlightDate'], origin_ratio['DelayRatio'], label=origin)
plt.xlabel('Flight Date')
plt.xticks(rotation=90)
plt.ylabel('Delay Ratio')
plt.title('Origin Delay Ratios by Date')
plt.legend()
plt.savefig('figures/DelayRatios_ATL_LAX_SFO.png')

# %% markdown
# ## observations
# There is certainly a chance that the delay ratios for SFO peaked after LAX and ATL later in the month.
# Earlier in the month the opposite occurs. However this could be due to delayed flights in SFO going to LAX and ATL.
# More testing is required and from here we will need a more automated analysis to test. 
