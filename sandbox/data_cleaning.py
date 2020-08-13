import numpy as np
import pandas as pd

cd_data = 'data/'
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

# cleanining my string of columns.
col = col.replace("\'", '')
col = col.replace('\n', '')
col = col.split(' ')

elements_to_remove = ['Unnamed:', '109']

for element in elements_to_remove:
    col.remove(element)

col.append('Unnamed: 109') # adding this column back in because it had a space and was split.

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


#Delays are big offenders in missing data. Mathmatically I need these missing values to be 0.
df['WeatherDelay'] = df['WeatherDelay'].fillna(0)
df['SecurityDelay'] = df['SecurityDelay'].fillna(0)
df['ArrDelay'] = df['ArrDelay'].fillna(0)
df['DepDelay'] = df['DepDelay'].fillna(0)
df['CarrierDelay'] = df['CarrierDelay'].fillna(0)
df['NASDelay'] = df['NASDelay'].fillna(0)

# Replacing  Nulls with mean values.


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
np.random.seed(1111)
df['LateAircraftDelay'] = df['LateAircraftDelay'].fillna(pd.Series(np.random.normal(df['LateAircraftDelay'])))
df['AirTime'] = df['AirTime'].fillna(pd.Series(np.random.normal(df['AirTime']))) # After plotting this I realized that this needs to be sampled from a normal distribution.

# Using Pandas statistics to get general information about the clean data.
print(df.info(),
'\n__Is Null Sum__\n', df.isnull().sum() ,
'\n__Describe__\n', df.describe().transpose())

# Saving clean data.
df.to_csv(cd_data+'flight_delays.csv', index=False)
