# %% markdown
# # Predicting Flight Delays
# %% markdown
# ### Problem Statement
# *What factors cause airline flight delays in commercial operations and can those factors be used to predict flight delays within 24 hours with an accuracy of at least 90% enabling air traffic to compensate and recover from said delays improving passenger's experience?*
#
#
# %% markdown
# ### Data Sources
# %% markdown
# The data used in this project is historical data from data.world using [flight data from across the US in January 2017](https://data.world/hoytick/2017-jan-ontimeflightdata-usa)
# %% markdown
# ### Choosing columns.
# While keeping the problem statement in mind, I have selected the columns that will be used to find causalities in flight delays. I hypothesize, that the most flight delays are due to a chain reaction from one another and originating from a root delay.
#
#
# %% markdown
# # Summary
#
# 1. **DepTime:** Actual Departure Time (local time: hhmm)
#
# 2. **TaxiOut:** Taxi Out Time, in Minutes
#
# 3. **TaxiIn:** Taxi In Time, in Minutes
#
# 4. **ArrTime:** Actual Arrival Time (local time: hhmm)
#
# 5. **Cancelled:** Cancelled Flight Indicator (1=Yes)
#
# 6. **Diverted:** Diverted Flight Indicator (1=Yes)
#
# 7. **AirTime:** Flight Time, in Minutes
#
# 8. **Distance:** Distance between airports (miles)
#
# 9. **WeatherDelay:** Weather Delay, in Minutes
#
# 10. **SecurityDelay:** Security Delay, in Minutes
#
# 11. **UniqueCarrier:** Unique Carrier Code. When the same code has been used by multiple carriers, a numeric suffix is used for earlier users, for example, PA, PA(1), PA(2). Use this field for analysis across a range of years.
#
# 12. ***OriginWac:** Origin Airport, World Area Code
#
# 13. **DestWac:** Destination Airport, World Area Code
#
# A list of World Area Codes can be found here: https://en.wikipedia.org/wiki/World_Area_Codes
#
# %% markdown
# Data Cleaning
# %% codecell
# __Dependencies__
import numpy as np
np.random.seed(1111)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# %% codecell
# __Variable Library__
cd_data = 'data/'
cd_figures = 'figures/'
df = pd.read_csv(cd_data+'On_Time_On_Time_Performance_2017_1.csv', low_memory=False)

col = ("""'Year' 'Quarter' 'Month' 'DayofMonth' 'DayOfWeek' 'FlightDate'
 'UniqueCarrier' 'AirlineID' 'Carrier' 'TailNum' 'FlightNum'
 'OriginAirportID' 'OriginAirportSeqID' 'OriginCityMarketID' 'Origin'
 'OriginCityName' 'OriginState' 'OriginStateFips' 'OriginStateName'
 'OriginWac' 'DestAirportID' 'DestAirportSeqID' 'DestCityMarketID' 'Dest'
 'DestCityName' 'DestState' 'DestStateFips' 'DestStateName' 'DestWac'
 'CRSDepTime' 'DepTime' 'DepDelay' 'DepDelayMinutes' 'DepDel15'
 'DepartureDelayGroups' 'DepTimeBlk' 'TaxiOut' 'WheelsOff' 'WheelsOn'
 'TaxiIn' 'CRSArrTime' 'ArrTime' 'ArrDelay' 'ArrDelayMinutes' 'ArrDel15'
 'ArrivalDelayGroups' 'ArrTimeBlk' 'Cancelled' 'CancellationCode'
 'Diverted' 'CRSElapsedTime' 'ActualElapsedTime' 'AirTime' 'Flights'
 'Distance' 'DistanceGroup' 'CarrierDelay' 'WeatherDelay' 'NASDelay'
 'SecurityDelay' 'LateAircraftDelay' 'FirstDepTime' 'TotalAddGTime'
 'LongestAddGTime' 'DivAirportLandings' 'DivReachedDest'
 'DivActualElapsedTime' 'DivArrDelay' 'DivDistance' 'Div1Airport'
 'Div1AirportID' 'Div1AirportSeqID' 'Div1WheelsOn' 'Div1TotalGTime'
 'Div1LongestGTime' 'Div1WheelsOff' 'Div1TailNum' 'Div2Airport'
 'Div2AirportID' 'Div2AirportSeqID' 'Div2WheelsOn' 'Div2TotalGTime'
 'Div2LongestGTime' 'Div2WheelsOff' 'Div2TailNum' 'Div3Airport'
 'Div3AirportID' 'Div3AirportSeqID' 'Div3WheelsOn' 'Div3TotalGTime'
 'Div3LongestGTime' 'Div3WheelsOff' 'Div3TailNum' 'Div4Airport'
 'Div4AirportID' 'Div4AirportSeqID' 'Div4WheelsOn' 'Div4TotalGTime'
 'Div4LongestGTime' 'Div4WheelsOff' 'Div4TailNum' 'Div5Airport'
 'Div5AirportID' 'Div5AirportSeqID' 'Div5WheelsOn' 'Div5TotalGTime'
 'Div5LongestGTime' 'Div5WheelsOff' 'Div5TailNum' 'Unnamed: 109'""")
# cleaning string of column names.
# __Data Cleaning__
col = col.replace("\'", '')
col = col.replace('\n', '')
col = col.split(' ')
elements_to_remove = ['Unnamed:', '109']
for element in elements_to_remove:
    col.remove(element)
col.append('Unnamed: 109') # adding this column back in because it had a space and was split into two elements. It will be removed later.

# These are the columns chosen for analysis.
keep_col = [
    'UniqueCarrier',
    'ArrTime',
    'DepTime',
    'Distance',
    'WeatherDelay',
    'Diverted',
    'TaxiIn',
    'TaxiOut',
    'Cancellation',
    'AirTime',
    'SecurityDelay',
    'DestWac',
    'OriginWac',
    'ArrDelay',
    'DepDelay',
    'CarrierDelay',
    'LateAircraftDelay',
    'NASDelay',
    'Origin',
    'FlightDate',
    'Dest']

col_names = set(col).intersection(set(keep_col))
remove_col = col
df = df[col_names]

#Delays are big offenders in missing data. Mathematically, I need these missing values to be 0.
df['WeatherDelay'] = df['WeatherDelay'].fillna(0)
df['SecurityDelay'] = df['SecurityDelay'].fillna(0)
df['ArrDelay'] = df['ArrDelay'].fillna(0)
df['DepDelay'] = df['DepDelay'].fillna(0)
df['CarrierDelay'] = df['CarrierDelay'].fillna(0)
df['NASDelay'] = df['NASDelay'].fillna(0)

# Logically, it does not make sense to replace missing values for Departure and arrival time with 0. These will be replaced with mean values.
DepTime_mean = df['DepTime'].mean()
df['DepTime'] = df['DepTime'].fillna(DepTime_mean)
ArrTime_mean = df['ArrTime'].mean()
df['ArrTime'] = df['ArrTime'].fillna(ArrTime_mean)

# We can now do the same thing for Taxi Time and AirTime.
TaxiIn_mean = df['TaxiIn'].mean()
df['TaxiIn'] = df['TaxiIn'].fillna(TaxiIn_mean)
TaxiOut_mean = df['TaxiOut'].mean()
df['TaxiOut'] = df['TaxiOut'].fillna(TaxiOut_mean)
TaxiIn_mean = df['AirTime'].mean()

# After plotting this I realized that this needs to be sampled from a normal distribution. Replacing these values with the mean, creates false trends in our analysis.
df['LateAircraftDelay'] = df['LateAircraftDelay'].fillna(pd.Series(np.random.normal(df['LateAircraftDelay'])))
df['AirTime'] = df['AirTime'].fillna(pd.Series(np.random.normal(df['AirTime']))) #
# %% codecell
# Using Pandas statistics to get general information about the clean data.
print(df.info(),
'\n__Is Null Sum__\n', df.isnull().sum() ,
'\n__Describe__\n', df.describe().transpose())

# %% codecell
# Saving clean data.
df.to_csv(cd_data+'flight_delays.csv', index=False)
# %% markdown
# ### Exploratory Data Analysis
# %% codecell
file = 'flight_delays.csv'
df = pd.read_csv(cd_data+file)
df.head()

# %% codecell
df_numbers_only = df.drop('UniqueCarrier', axis=1)
sns.heatmap(df_numbers_only.corr())
plt.savefig(cd_figures+'Heatmap.png')

# %% markdown
# ## Heatmap Observations
#  - Wac shows a correlation, however this is an ID according to Column_info.txt and can be igonored.
#  - There is a correlation between departure time and arrival time that is nearly common sense, however further investigation is certainly warrented.
#  - The distance and air time columns have a strong correlation. This is not all that surprising, however this can be investigated with other parameters.
#  - Departure delay and late aircraft delay is correlated. Likely an easy prediciton once the aircraft delay is accounced for but we are looking to predict the initial delay.
#  - ~~A diverted aircraft lines up with late aicraft delays, arrival delays, carrier delays, and NAS delays. According to the data set's information the NAS is "National Air System Delay, in Minutes".~~
#     - After resampling late aircraft delays and airtime, these correlations did not appear in the heat map.
#  - The departure and late aircraft correlation is suprisingly weak.
#  What we are really interested in here is the delays and other causes that can contribute to these recorded delays.

# %% codecell
plt.figure(figsize=(10,10))
plt.scatter(df['DepTime'], df['ArrTime'], alpha=0.1, color='blue', s=0.03)
plt.title('Departure Time VS Arrival Time')
plt.xlabel('Departure Time')
plt.ylabel('Arrival Time')
plt.savefig(cd__figures+'DepTime_VS_ArrTime.png')

# %% markdown
# As expected there is a very strong correlation with departure time and arrival time. This could be clearly predicted with a linear regression model and certainly could be a piece in later analysis.

# %% codecell
plt.figure(figsize=(10,10))
plt.scatter(df['Distance'], df['AirTime'], alpha=0.1, color='blue', s=0.8)
plt.title('Distance VS Air Time')
plt.xlabel('Distance (Miles)')
plt.ylabel('Air Time (Minutes)')
plt.savefig(cd_figures+'Distance_VS_AirTime.png')
# %% markdown
# The linear regression is even more pronounced here showing a narrow range of posibilities.
#
# ~~There seems to also be a straight line hovering above 100 minutes of airtime with a range of distance. This could mean that there is an issue in the reported data. Likely where I filled Nan values with Means.~~
# *(This was due to Nans being replaced by means. Since then they have been resampled at the line has been removed.)*

# %% codecell
plt.figure(figsize=(10,10))
plt.scatter(df['DepDelay'], df['LateAircraftDelay'], alpha=0.5, color='blue', )
plt.title('Departure Delay VS Late Aircraft')
plt.xlabel('Departure Delay (Minutes)')
plt.ylabel('Late Aircraft (Minutes)')
plt.savefig(cd_figures+'DepDelay_VS_LateAircraft.png')

# %% markdown
# Again there is a sharp linear trend with most data points sitting below 200 minutes.
# Several outliers sit well beyond that mark but are still within the regression.
# Many data points are showing a late aircraft delay of 0. Which is remarkable even after resampling missing data.
# This suggests that departure delays are not always a causation of a late aircraft.

# %% markdown
# ## Scatterplot Observations
# Now that I have had some observations on the measurements on flight delays, I can start attacking the date time objects and looking for when one flight delay that will cause another.
# My hypothosis is: " The amount of flights delayed at the origin will affect the amount of flights delayed at the destination."
# In order to test my hypthosis, I will need to narrow down on one airport and find some flight delays. Then I can follow those flights and look for these chain reaction delays causing other flights to be delayed.

# %% codecell
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
destinations = list(df['Dest'].unique())
groups_count = {}
for destination in destinations:
    groups_count[destination] = df[(df['Dest'] == destination) & (df['DelaySum'] > 0)].count()

df_group = pd.DataFrame(groups_count).max()
df_group[df_group == df_group.max()]
# %% markdown
# This shows that the Los Angelos International Airport was the biggest offender of incoming flight delays.
# This may be prosecuter's fallacy because LAX happens to also be one of the busiest airports.
# Either way it makes a good place to start.
#
# From here the experiement will work as follows:
# - Treat a flight delay like it has a half life and count the number of relational delays going out of LAX to the next similar airport.
# - Then we can expect that there is a chance this amount of delays will cause more delays at the next airport.
#
# %% codecell
destinations = list(df['Origin'].unique())
groups_count = {}
for destination in destinations:
    groups_count[destination] = df[(df['Origin'] == destination) & (df['DelaySum'] > 0)].count()

df_group = pd.DataFrame(groups_count).max()
df_group[df_group == df_group.max()]
# %% markdown
# Using the same methods, this shows that Hartsfield-Jackson Atlanta International Airport is the biggest offender in January 2017 of out going flight delays.

# %% codecell
dates = list(df['FlightDate'].unique())
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
            # If there are no flights at an airport. There is a possibility to divide by 0
            delay_ratio = 0

        result['Origin'].append(origin)
        result['FlightDate'].append(date)
        result['DelayRatio'].append(delay_ratio)
    return(pd.DataFrame(result))

# %% codecell

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
plt.savefig(cd_figures+'DelayRatios.png')

# %% markdown
# ### Testing
#
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
plt.savefig(cd_figures+'DelayRatios_ATL_LAX_SFO.png')
# %% markdown
# ## Observations
# There is certainly a chance that the delay ratios for SFO peaked after LAX and ATL later in the month.
# Earlier in the month the opposite occurs. However this could be due to delayed flights in SFO going to LAX and ATL.
# More testing is required to be sure.
# %% markdown
# From here we will need to split the data into regions and find the mean delay ratios per region. First I will need to define regions and airports within those regions. I will use the this list of [Top 100 Airports](http://www.fi-aeroweb.com/Top-100-US-Airports.html). Then we can divide the airports into regions according to the [Census Region Division of the US](https://www2.census.gov/geo/pdfs/maps-data/maps/reference/us_regdiv.pdf)
# %% markdown
# Selecting the airport identifiers and assigning the a variables by region.
# %% codecell
pacific = ['LAX', 'SFO', 'SEA', 'SAN', 'HNL', 'PDX', 'OAK', 'SJC', 'SMF', 'SNA', 'OGG', 'ANC', 'BUR', 'ONT', 'LGB', 'KOA', 'GEG', 'LIH', 'PSP', ]
mountain = ['DEN', 'LAS', 'PHX', 'SLC', 'ABQ', 'RNO', 'BOI', 'TUS', 'COS']
west_north_central = ['DFW', 'MSP', 'STL', 'MCI', 'NE', 'DSM', 'ICT']
west_south_central = ['IAH', 'DAL', 'AUS', 'HOU', 'MSY', 'SAT', 'OKC', 'ELP', 'TUL', 'LIT', ]
east_north_central = ['ORD','DTW','MDW', 'CLE', 'IND', 'CVG', 'CMH', 'MKE', 'GRR', 'MSN', 'DAY']
east_south_central = ['BNA', 'MEM', 'SDF', 'BHM', 'TYS']
middle_atlantic = ['JFK', 'EWR', 'LGA', 'PHL', 'PIT', 'BUF', 'BHM', 'ROC', 'SYR']
south_atlantic = ['ATL', 'CLT', 'MCO', 'MIA', 'FLL', 'BWI', 'DCA', 'IAD', 'TPA', 'RDU', 'RSW', 'PBI', 'JAX', 'CHS', 'RIC', 'ORF', 'SFB', 'SAV', 'MYR', 'GSP', 'PIE', 'GSO', 'PNS']
new_england = ['BOS', 'BDL', 'PVD', 'MHT', 'PWM']
# %% markdown
# Seperating the DataFrame into regions.
# %% codecell
pacific_df = df[df['Origin'].isin(pacific)]
mountain_df = df[df['Origin'].isin(mountain)]
west_north_central_df = df[df['Origin'].isin(west_north_central)]
west_south_central_df = df[df['Origin'].isin(west_south_central)]
east_north_central_df = df[df['Origin'].isin(east_north_central)]
east_south_central_df = df[df['Origin'].isin(east_south_central)]
middle_atlantic_df = df[df['Origin'].isin(middle_atlantic)]
south_atlantic_df = df[df['Origin'].isin(south_atlantic)]
new_england_df = df[df['Origin'].isin(new_england)]
# %% markdown
# Calculating all ratios by region.
# %% codecell
def calculate_region_ratios(regions):
    """
    Accepts an iterator and loops through
    the calculate_origin_ratios funtion to
    generate a DataFrame of origin delay
    ratios by region.
    """
    ratios = []
    for region in regions:
        region_ratio = calculate_origin_ratios(region)
        ratios.append(region_ratio)
    return pd.concat(ratios)
# %% codecell
pacific_delays = calculate_region_ratios(pacific)
mountain_delays = calculate_region_ratios(mountain)
west_north_central_delays = calculate_region_ratios(west_north_central)
west_south_central_delays = calculate_region_ratios(west_south_central)
east_north_central_delays = calculate_region_ratios(east_north_central)
east_south_central_delays = calculate_region_ratios(east_south_central)
middle_atlantic_delays = calculate_region_ratios(middle_atlantic)
south_atlantic_delays = calculate_region_ratios(south_atlantic)
new_england_delays = calculate_region_ratios(new_england)

# %% markdown
# Calculating the mean by region.
# %% codecell
def calculate_region_means(region, df=df):
    """
    Takes the DataFrame "region" and calculates the mean of delay ratios by region.
    """
    flight_dates = region['FlightDate'].unique()
    ratio_means = {'FlightDate': [], 'DelayRatio': []}
    for flight_date in flight_dates:
        ratio_means['FlightDate'].append(flight_date)
        ratio_means['DelayRatio'].append(np.mean(region[region['FlightDate'] == flight_date]['DelayRatio']))
    return pd.DataFrame(ratio_means)

# %% codecell
pacific_DelayRatio = calculate_region_means(pacific_delays)
mountain_DelayRatio = calculate_region_means(mountain_delays)
west_north_central_DelayRatio = calculate_region_means(west_north_central_delays)
west_south_central_DelayRatio = calculate_region_means(west_south_central_delays)
east_north_central_DelayRatio = calculate_region_means(east_north_central_delays)
east_south_central_DelayRatio = calculate_region_means(east_south_central_delays)
middle_atlantic_DelayRatio = calculate_region_means(middle_atlantic_delays)
south_atlantic_DelayRatio = calculate_region_means(south_atlantic_delays)
new_england_DelayRatio = calculate_region_means(new_england_delays)
# %% markdown
# Plotting the region means to search for trends in the data.
# %% codecell
plt.figure(figsize=(20,10))
plt.plot(pacific_DelayRatio['FlightDate'], pacific_DelayRatio['DelayRatio'], label='Pacific')
plt.plot(mountain_DelayRatio['FlightDate'], mountain_DelayRatio['DelayRatio'], label='Mountain')
plt.plot(west_north_central_DelayRatio['FlightDate'], west_north_central_DelayRatio['DelayRatio'], label='West North Central')
plt.plot(west_south_central_DelayRatio['FlightDate'], west_south_central_DelayRatio['DelayRatio'], label='West South Central')
plt.plot(east_north_central_DelayRatio['FlightDate'], east_north_central_DelayRatio['DelayRatio'], label='East North Central')
plt.plot(east_south_central_DelayRatio['FlightDate'], east_south_central_DelayRatio['DelayRatio'], label='East South Central')
plt.plot(middle_atlantic_DelayRatio['FlightDate'], middle_atlantic_DelayRatio['DelayRatio'], label='Middle Atlantic')
plt.plot(south_atlantic_DelayRatio['FlightDate'], south_atlantic_DelayRatio['DelayRatio'], label='South Atlantic')
plt.plot(new_england_DelayRatio['FlightDate'], new_england_DelayRatio['DelayRatio'], label='New England')
plt.xlabel('Flight Date')
plt.xticks(rotation=90)
plt.ylabel('Delay Ratio')
plt.legend()
plt.grid()
plt.title('Delay Ratios by Date')
plt.savefig(cd_figures+'DelayRatios_all_regions.png')
# %% markdown
# ## Observations
# Flight delays across the US seem to be closely related. I would assume weather is a factor in the uniformity, however delays of the same day tend to have a peak with waves of peaks at other airports. This plot supports the theory that we can measure the flight delays based off of delay ratios by region.
# %% markdown
# # Processing for Machine Learning
# We are ready to prepare our data for a model. I will make use of one-hot encoding for classification columns like the airport identifier and the region. Then change the flight dates to an integer. And lastly we can leave the flight ratio as is beacuse we have already normalized this information manually. This process will ensure the data is healthy enough to train our algorthim.
#
# Then we can use skealrn's train_test_split function to split the data into training and testing sets.
# %% codecell
from sklearn.model_selection import train_test_split
# %% codecell
pacific_delays['Region'] = 'Pacific'
mountain_delays['Region'] = 'Mountain'
west_north_central_delays['Region'] = 'WestNorthCentral'
west_south_central_delays['Region'] = 'WestSouthCentral'
east_north_central_delays['Region'] = 'EastNorthCentral'
east_south_central_delays['Region'] = 'EastSouthCentral'
middle_atlantic_delays['Region'] = 'MiddleAtlantic'
south_atlantic_delays['Region'] = 'SouthAtlantic'
new_england_delays['Region'] = 'NewEngland'

# %% codecell
df = pd.concat([
    pacific_delays,
    mountain_delays,
    west_north_central_delays,
    west_south_central_delays,
    east_north_central_delays,
    east_south_central_delays,
    middle_atlantic_delays,
    south_atlantic_delays,
    new_england_delays])
# %% codecell
import datetime as dt
regions = [
    pacific_delays,
    mountain_delays,
    west_north_central_delays,
    west_south_central_delays,
    east_north_central_delays,
    east_south_central_delays,
    middle_atlantic_delays,
    south_atlantic_delays,
    new_england_delays]

for region in regions:
    region['FlightDate'] = pd.to_datetime(region['FlightDate'])
    region['FlightDate'] = region['FlightDate'].dt.day.apply(int)

    X = region.drop('DelayRatio', axis=1)
    y = region[['DelayRatio']]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1111)

    X_train.to_csv(cd_data+'X_train_'+region.at[0,'Region'][0]+'.csv', index=False)
    X_test.to_csv(cd_data+'X_test_'+region.at[0,'Region'][0]+'.csv', index=False)
    y_train.to_csv(cd_data+'y_train_'+region.at[0,'Region'][0]+'.csv', index=False)
    y_test.to_csv(cd_data+'y_test_'+region.at[0,'Region'][0]+'.csv', index=False)
# %% codecell
    pacific_delays['FlightDate'] = pd.to_datetime(pacific_delays['FlightDate'])
    pacific_delays['FlightDate'] = pacific_delays['FlightDate'].dt.day
# %% codecell
# df ['DelayRatio'] = df['DelayRatio'].fillna(0)
# oh = pd.get_dummies(df) # Skhttps://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.at.html?highlight=#pandas.DataFrame.atipping this after mentor consultation.
# %% codecell
df.info()
# %% codecell
X = df.drop('DelayRatio', axis=1)
y = df[['DelayRatio']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1111)
# %% markdown
# ## Writing the Data
# Writing our data back into a .csv.
# %% codecell
X_train.to_csv(cd_data+'X_train.csv', index=False)
X_test.to_csv(cd_data+'X_test.csv', index=False)
y_train.to_csv(cd_data+'y_train.csv', index=False)
y_test.to_csv(cd_data+'y_test.csv', index=False)

# %% markdown
# # Choosing a model.
# ### Linear Regression
# First, running the experiment on linear regression using Scikit Learn's API.
# %% codecell
# Linear Regression Model Expiriment
# __Dependancies__
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
# __Variables__
cd_data = 'data/'
regions = ['Pacific',
'Mountain',
'WestNorthCentral',
'WestSouthCentral',
'EastNorthCentral',
'EastSouthCentral',
'MiddleAtlantic',
'SouthAtlantic',
'NewEngland']

# Running and plotting the model
plt.figure(figsize=(10,10))
def predict_delays_by_region(regions):
    """
    Linear Regression model that loads in multiple regions
    of data and plots all predicitons on the same plot.
    """
    for region in regions:
        # Load in the data
        X_train = pd.read_csv(cd_data+'X_train_'+region+'.csv')
        X_test = pd.read_csv(cd_data+'X_test_'+region+'.csv')
        y_train = pd.read_csv(cd_data+'y_train_'+region+'.csv')
        y_test = pd.read_csv(cd_data+'y_test_'+region+'.csv')
        # Replace hidden Nan's that may have leaked into the data.
        y_train['DelayRatio'].fillna(np.mean(y_train['DelayRatio']), inplace=True)
        y_test['DelayRatio'].fillna(np.mean(y_test['DelayRatio']), inplace=True)
        # Creating and fitting the model
        lr = LinearRegression()
        lr.fit(X_train[['FlightDate']], y_train)
        y_pred = lr.predict(X_test[['FlightDate']])
        # Plotting the results.
        plt.scatter(X_test['FlightDate'], y_test, label=region, alpha=0.6, s=10)
        plt.plot(X_test[['FlightDate']], y_pred, color='red')
        plt.legend()
    plt.title('LinearRegression on Airport Flight Delays By Region')
    plt.xlabel('Day in January, 2017')
    plt.ylabel('Delay Ratio')
    plt.savefig('figures/LinearRegression_on_all_flight_delay_ratios.png')

# Running the model function.
predict_delays_by_region(regions)

# %% markdown
# Predictions are shown in red. This model shows a trend that predicts all flight delays will end in the next few months.Since this the probability of flight delays ending and actually reversing seems unlikely, we can assume not to trust this model.
#
# Due to the spread of the data, I am concerned that any regression model will be able to make useful predicitons. Next I will try to classify the delays with a random forest.
# %% markdown
# ## RandomForest
# %% codecell
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
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
print('Confusion Matrix: \n'+str(confusion_matrix(y_test_bi, y_pred)))
print('Accuracy Score: \n'+str(model.score(X_test, y_test_bi)))
plt.figure(figsize=(10,10))
plt.scatter(X_test['FlightDate'], y_test, c=y_pred)

plt.title('RandomForestClassifier delays < 15%')
plt.xlabel('Day of Flight Date in Januwary 2017')
plt.ylabel('Delay Ratio')
plt.legend()
plt.savefig('figures/RandomForestClassifier_FlightDelays.png')

# %% markdown
# Our prediction is giving okay metrics of nearly 75% accuracy. Still, this does not satisfy our accuracy requirements. Even worse is we can assume this model is over-fit to our data and will make even worse predictions moving forward as the seasons change and flights are affected differently from weather. This was an interesting experiment, however we need to get regression to work.
#
# Next I will try an SVR model. I think this makes sense because it will be better suited for all the outliers we are currently struggling to work with.
#
# %% markdown
# # SVR
# %% codecell
# Support Vecter Machine Expiriment
# __Dependancies__
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error
# __Variables__
cd_data = 'data/'
regions = ['Pacific',
'Mountain',
'WestNorthCentral',
'WestSouthCentral',
'EastNorthCentral',
'EastSouthCentral',
'MiddleAtlantic',
'SouthAtlantic',
'NewEngland']
# figure being created outside of loop to ensure there will only be one plot.
plt.figure(figsize=(10,10))
def predict_delays_by_region(regions):
    """
    SVM model that loads in multiple regions
    of data and plots all predicitons on the same plot.
    """
    for region in regions:
        # Load in the data
        X_train = pd.read_csv(cd_data+'X_train_'+region+'.csv')
        X_test = pd.read_csv(cd_data+'X_test_'+region+'.csv')
        y_train = pd.read_csv(cd_data+'y_train_'+region+'.csv')
        y_test = pd.read_csv(cd_data+'y_test_'+region+'.csv')
        # Replace hidden Nan's that may have leaked into the data.
        y_train['DelayRatio'].fillna(np.mean(y_train['DelayRatio']), inplace=True)
        y_test['DelayRatio'].fillna(np.mean(y_test['DelayRatio']), inplace=True)
        # Creating and fitting the model
        model = SVR(kernel='rbf', gamma=0.09, C=0.65)
        model.fit(X_train[['FlightDate']], y_train)
        y_pred = model.predict(X_test[['FlightDate']])
        # Plotting the results.
        plt.scatter(X_test['FlightDate'], y_test, label=region, alpha=0.6, s=10)
        plt.scatter(X_test[['FlightDate']], y_pred, color='red')
        plt.legend()
        plt.grid()
    plt.title('SVR on Flight Delays')
    plt.xlabel('Day in January, 2017')
    plt.ylabel('Delay Ratio')
    plt.xticks(range(1,32))
    plt.savefig('figures/SVR_flightDelay_prediciton.png')
# Running the model function.
predict_delays_by_region(regions)

# %% markdown
# This is the model I would use for now. However, I am finding a bias in the time constraint.
# These flights are affected by seasonal storms. Which means my model will need to be continuously trained on more data for more accurate results.
# Here we look at flights only in January 2017, however if we continue to observe flights over several years, I suspect we will build a better prediction model.
#
# For now I will break these up into predictions by region and compute the MSE and MAE.
# %% codecell
# Support Vecter Machine by region
# __Dependancies__
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error
# __Variables__
regions = ['Pacific',
'Mountain',
'WestNorthCentral',
'WestSouthCentral',
'EastNorthCentral',
'EastSouthCentral',
'MiddleAtlantic',
'SouthAtlantic',
'NewEngland']

def predict_delays_by_region(regions):
    """
    SVM model that loads in multiple regions
    of data and plots all predicitons on seperate plots
    by region.
    """
    for region in regions:
        # Load in the data
        X_train = pd.read_csv(cd_data+'X_train_'+region+'.csv')
        X_test = pd.read_csv(cd_data+'X_test_'+region+'.csv')
        y_train = pd.read_csv(cd_data+'y_train_'+region+'.csv')
        y_test = pd.read_csv(cd_data+'y_test_'+region+'.csv')
        # Replace hidden Nan's
        y_train['DelayRatio'].fillna(np.mean(y_train['DelayRatio']), inplace=True)
        y_test['DelayRatio'].fillna(np.mean(y_test['DelayRatio']), inplace=True)
        # Creating and fitting the model
        model = SVR(kernel='rbf', gamma=0.09, C=0.65)
        plt.figure(figsize=(10,10))
        model.fit(X_train[['FlightDate']], y_train)
        y_pred = model.predict(X_test[['FlightDate']])
        # Plotting the results.
        plt.scatter(X_test['FlightDate'], y_test, label=region, alpha=0.6)
        plt.scatter(X_test[['FlightDate']], y_pred, color='red', label='Prediction')
        plt.legend()
        plt.grid()
        plt.text(x=1, y=max(y_test['DelayRatio']/40), s='MSE: '+
        str(round(mean_squared_error(y_test, y_pred), 3))+
        '\nMAE: '+ str(round(mean_absolute_error(y_test, y_pred), 3)))
        plt.title('SVR on Flight Delays')
        plt.xlabel('Day in January, 2017')
        plt.ylabel('Delay Ratio')
        plt.xticks(range(1,32))
        plt.savefig('figures/SVR_on_'+region+'flightDelay_prediciton.png')
# Running the model function.
predict_delays_by_region(regions)
