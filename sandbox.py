import pandas as pd

regions = ['Pacific',
'Mountain',
'WestNorthCentral',
'WestSouthCentral',
'EastNorthCentral',
'EastSouthCentral',
'MiddleAtlantic',
'SouthAtlantic',
'NewEngland']

cd_data = 'data/'
for region in regions:
    X_train = pd.read_csv(cd_data+'X_train_'+region+'.csv')
    X_test = pd.read_csv(cd_data+'X_test_'+region+'.csv')
    y_train = pd.read_csv(cd_data+'y_train_'+region+'.csv')
    y_test = pd.read_csv(cd_data+'y_test_'+region+'.csv')
    print(region+'____\n', X_train.isna().sum(), X_test.isna().sum(), y_train.isna().sum(), y_test.isna().sum())
