import numpy as np
import pandas as pd

path = '../data/raw/On_Time_On_Time_Performance_2017_1.csv'
df = pd.read_csv(path, low_memory=False)


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

col = col.replace("\'", '')
col = col.replace('\n', '')
col = col.split(' ')

elements_to_remove = ['Unnamed:', '109']

for element in elements_to_remove:
    col.remove(element)

col.append('Unnamed: 109')

# These are the columns chosen for analysis.
keep_col = ['UniqueCarrier', 'ArrTime', 'DepTime', 'OriginAirportID', 'DestAirportID', 'Distance', 'WeatherDelay', 'Diverted', 'TaxiIn', 'TaxiOut', 'Cancellation']

col_names = set(col).intersection(set(keep_col))
col_names



df['UniqueCarrier'].value_counts()
df['OriginAirportID'].value_counts()
df['DestAirportID'].value_counts()
df[['ArrTime','DepTime']].info()
df['Distance'].describe() #Distance between airports in Miles.
df['WeatherDelay'].describe()
df['Diverted'].describe()
df[['TaxiIn', 'TaxiOut']].corr().describe()

remove_col = col

df = df[col_names]
#WeatherDelay is a big offender in missing data. Mathmatically I need these missing values to be 0.
df['WeatherDelay'] = df['WeatherDelay'].fillna(0)
# Replacing DepTime and ArrTime Nulls with mean values.

DepTime_mean = df['DepTime'].mean()
df['DepTime'] = df['DepTime'].fillna(DepTime_mean)
ArrTime_mean = df['ArrTime'].mean()
df['ArrTime'] = df['ArrTime'].fillna(ArrTime_mean)

# We can now do the same thing for Taxi Time.

TaxiIn_mean = df['TaxiIn'].mean()
df['TaxiIn'] = df['TaxiIn'].fillna(TaxiIn_mean)
TaxiOut_mean = df['TaxiOut'].mean()
df['TaxiOut'] = df['TaxiOut'].fillna(TaxiOut_mean)

df.info()
df.isnull().sum()
df.head()
df.describe()

df.to_csv('../data/interim/flight_delays.csv')
