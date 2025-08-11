from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.utils import resample
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error

# Importing the Data from xlsx file
Data = np.array(pd.read_excel(r'C:\Users\Trevo\Documents\CFD_Surrogate_Data.xlsx'))

# Categorizing the data into the regressor (y) and the features (X)
y = Data[:, 0]
X = Data[:, 1:5]

# Splitting the data into the testing and training sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Creating a range for the applicable hyperparameters needed for hypertuning
parameters = {
    'n_estimators': np.arange(250, 400, 1),
    'learning_rate': np.arange(0.2, 0.3, 0.01),
    'max_depth': np.arange(1, 12,1)
}

# Creating the Gradient Boosting Regressor
regressor = GradientBoostingRegressor()

# Using GridSearchCV for an exhaustive search through all hyperparameters
tuning_model = GridSearchCV(regressor, param_grid=parameters, scoring='neg_mean_squared_error', cv=5, verbose=3, n_jobs=-1)
tuning_model.fit(X_train, y_train)

# Extracting the best hyperparameters from the tuning_model
best_model = tuning_model.best_estimator_
print(best_model)
# Bootstrapping: Using multiple samples to fit and predict
n_bootstrap_samples =365 # Number of bootstrap samples
y_pred_train_bootstrap=np.zeros((n_bootstrap_samples,len(X_train)))
y_pred_bootstrap = np.zeros((n_bootstrap_samples, len(X_test)))

# Bootstrap sampling and fitting models
for i in range(n_bootstrap_samples):
    # Generate a bootstrap sample
    X_train_bootstrap, y_train_bootstrap = resample(X_train, y_train, random_state=i)
    
    # Fit the model using the best hyperparameters to the bootstrap sample
    best_model.fit(X_train_bootstrap, y_train_bootstrap)
    
    # Predict on the test set and store the results
    y_pred_train_bootstrap[i, :] = best_model.predict(X_train)
    y_pred_bootstrap[i, :] = best_model.predict(X_test)

# Averaging the predictions from all bootstrap models
y_pred_train = np.mean(y_pred_train_bootstrap,axis=0)
y_pred = np.mean(y_pred_bootstrap, axis=0)

# Calculating the relative percentage error
percentage_error = []
for i in range(0, len(y_pred)):
    if y_test[i] != 0:  # Avoid division by zero
        percentage_error.append(100 * (y_pred[i] - y_test[i]) / y_test[i])
    else:
        percentage_error.append(np.nan)  # Set as NaN or handle as needed

# Calculate the accuracy metrics
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

# Print the results
print(f'R^2 Score: {r2}')
print(f'Mean Absolute Error: {mae}')

# Plotting the True and Predicted regressor for the test data
x = np.arange(0, len(X_test))
plt.figure(1)
plt.scatter(x, np.transpose(y_test), label="True")
plt.scatter(x, np.transpose(y_pred), label="Predicted")
plt.legend()
plt.ylabel('H (W/m2 K)')
plt.xlabel('Test Data')
plt.title('Gradient Boosting with Bootstrapping')

# Plotting the accuracy of the predicted regressors
plt.figure(2)
plt.scatter(x, percentage_error)
plt.xlabel('Test Data')
plt.ylabel('Error (%)')
plt.title('Gradient Boosting Error with Bootstrapping')

plt.show()


'''
parameters = {
    'n_estimators': np.arange(370, 382, 1),
    'learning_rate': np.arange(0.2, 0.22, 0.01),
    'max_depth': np.arange(1, 12,1)
}
'''