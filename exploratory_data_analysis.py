# %% codecell
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

# %% codecell
cd_data = 'data/'
file = 'flight_delays.csv'
df = pd.read_csv(cd_data+file)
df.head()
# %% codecell
print('\n___________Shape___________\n', df.shape, '\n___________Info___________\n')
print( df.info())
print('\n___________Statistics___________\n', df.describe())

# %% codecell
df_numbers_only = df.drop('UniqueCarrier', axis=1)
sns.heatmap(df_numbers_only.corr())
plt.savefig('figures/Heatmap.png')

# %% markdown
#
# ## Heatmap Observations
#  - Wac shows a correlation, however this is an ID according to Column_info.txt and can be igonored.
#  - There is a correlation between departure time and arrival time that is nearly common sense, however further investigation is certainly warrented.
#  - The distance and air time columns have a strong correlation. This is not all that surprising, however this can be investigated with other parameters.
#  - Departure delay and late aircraft delay is correlated. Likely an easy prediciton once the aircraft delay is accounced for but we are looking to predict the initial delay.
#  - ~~A diverted aircraft lines up with late aicraft delays, arrival delays, carrier delays, and NAS delays. According to the data set's information the NAS is "National Air System Delay, in Minutes".~~
#     - After resampling late aircraft delays and airtime, these correlations did not appear in the heat map.
#  - The departure and late aircraft correlation is suprisingly weak.
#  What we are really interested in here is the delays and other causes that can contribute to these recorded delays.
#
#
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
# The linear regression is even more pronounced here showing a narrow range of posibilities.
#
# ~~There seems to also be a straight line hovering above 100 minutes of airtime with a range of distance. This could mean that there is an issue in the reported data. Likely where I filled Nan values with Means.~~
# *(This was due to Nans being replaced by means. Since then they have been resampled at the line has been removed.)*
#
#
# %% codecell
plt.figure(figsize=(10,10))
plt.scatter(df['DepDelay'], df['LateAircraftDelay'], alpha=0.5, color='blue', )
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
#
# %% markdown
# ## Scatterplot Observations
# Now that I have had some observations on the measurements on flight delays, I can start attacking the date time objects and looking for when one flight delay that will cause another.
# My hypothosis is: " The amount of flights delayed at the origin will affect the amount of flights delayed at the destination."
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
print("\n________The highest average delay time________\n", df[df['Dest'] == 'DTW']['DelaySum'].mean())
print("\n________Flight info on in Januwary 2017 at Elmira Corning Regional Airport________\n", df[df['Dest'] == 'DTW'][flight_info])
# Number of flight delays for any reason for flights going to Detroit Metropolitan Airport
print("\n________The count of total flight delays in January at Elmira Corning Regional Airport________\n", df[(df['Dest'] == 'ELM') & (df['DelaySum'] > 0)]['DelaySum'].count())
print("\n________Delay Ratio_______\n", df[(df['Dest'] == 'ELM') & (df['DelaySum'] > 0)]['DelaySum'].count()/df[df['Dest'] == 'ELM']['DelaySum'].count())

# %% markdown
# I supsect the reason that I found Elmira Corning Regional Airport is because it does not have many data points at all! Infact there are only 7 delays in total and almost 54% of the flights from this airport were delayed!
# The central limit theorem would suggest that I have found this smaller airport by chance. The one long flight delay is wildely skewing the data.
#
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
# This shows that Hartsfield-Jackson Atlanta International Airport is the biggest offender in January 2017 of out going flight delays.
#
# Next we can develop a way to measure flight delays by origin and date.
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
plt.savefig('figures/DelayRatios.png')
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
plt.savefig('figures/DelayRatios_ATL_LAX_SFO.png')

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
plt.savefig('figures/DelayRatios_all_regions.png')
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
