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
