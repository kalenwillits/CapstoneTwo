# %% markdown
# # Data cleaning
# #### Setting the enviorment
# %% codecell
!pwd
import numpy as np
import pandas as pd
from pandas_profiling import ProfileReport

path = 'data/raw/On_Time_On_Time_Performance_2017_1.csv'
df = pd.read_csv(path, low_memory=False)
# %% markdown
# # Choosing columns.
# While keeping the problem statement in mind, I have selected the columns that will be used to find causalities in flight delays. I hypthesize, that tht most flight delays are due to a chain reaction from one another and orginating from a root delay.
#
# I will be leaving out coulumns that can be calculated from other columns to save on space and readability.
#
# ## Problem Statment
# *What factors cause airline flight delays in commercial operations and can those factors be used to predict flight delays within 24 hours with an accuracy of at least 90% enabling air traffic to compensate and recover from said delays improving passenger's experience?*
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
#
#
# %% codecell
#Which values are factors causing a delay?

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

# cleanining my string of columns.
col = col.replace("\'", '')
col = col.replace('\n', '')
col = col.split(' ')

elements_to_remove = ['Unnamed:', '109']

for element in elements_to_remove:
    col.remove(element)

col.append('Unnamed: 109') # adding this column back in because it had a space and was split.


# %% markdown
# # Removing Columns
# %% codecell
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
col_names

remove_col = col

df = df[col_names]

# %% markdown
# # Replacing NaN values
# Based on the type of data, I have determined how to handle missing values below.
# %% codecell
#Delays are big offenders in missing data. Mathmatically I need these missing values to be 0.
df['WeatherDelay'] = df['WeatherDelay'].fillna(0)

df['SecurityDelay'] = df['SecurityDelay'].fillna(0)

df['ArrDelay'] = df['ArrDelay'].fillna(0)

df['DepDelay'] = df['DepDelay'].fillna(0)

df['CarrierDelay'] = df['CarrierDelay'].fillna(0)


df['NASDelay'] = df['NASDelay'].fillna(0)




# Replacing DepTime and ArrTime Nulls with mean values.

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

# These are values that needed to be sampled to replace missing data.
df['LateAircraftDelay'] = df['LateAircraftDelay'].fillna(pd.Series(np.random.normal(df['LateAircraftDelay'])))
df['AirTime'] = df['AirTime'].fillna(pd.Series(np.random.normal(df['AirTime']))) # After plotting this I realized that this needs to be sampled from a normal distribution.

# %% codecell
len(keep_col) #checking length so I know I did not miss any columns.
# %% markdown
# # Exploring the Data
# Using Pandas statistics to get general information about the clean data.
# %% codecell
df.info()
# %% codecell
df.isnull().sum() #Checking for NaNs.
# %% codecell
df.head()
# %% codecell
df.describe()
# %% codecell
df.corr()
# %% markdown
# # Deep Exploration
# %% codecell
#profile = ProfileReport(df, title='Pandas Profiling Report', explorative=True)
#profile
# %% codecell
df.to_csv('data/interim/flight_delays.csv', index=False) # Saving clean data.
