# %% markdown
# # Choosing a model.
# First, running the experiment on linear regression using Scikit Learn's API.
# %% codecell
!python LinearRegression.py
# %% markdown
# [Linear Regression By region](LinearRegression_on_all_flight_delay_ratios.png)
# Predictions are shown in red. This model shows a trend that predicts all flight delays will end in the next few months.
# Since this the probability of flight delays ending and actually reversing seems unlikely, we can assume not to trust this model.
# %% markdown
# # Next I will try a support vector regression.
# %% codecell
!python svm.py
# %% markdown
# [Support Vector Regression](SVR_flightDelay_prediciton.png)
#  This model is far more promising. The MSE and MAE are showing us a better predictive model.
# %% markdown
# Ultimately, this is the model I would use for now. However, I am finding a bias in the time constraint.
# These flights are affected by seasonal storms. Which means my model will need to be continuously trained on more data for more accurate results.
# Here we look at flights only in January 2017, however if we continue to observe flights over several years, I suspect we will build a better prediction model.
