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
