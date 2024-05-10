import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler 
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score


# Importing the dataset
dataset = pd.read_csv('insurance.csv')
# convert data set to a pandas dataframe
df = pd.DataFrame(dataset)

# Encoding categorical data values
le = LabelEncoder()
le.fit(df.sex.drop_duplicates())
df.sex = le.transform(df.sex)

le.fit(df.smoker.drop_duplicates())
df.smoker = le.transform(df.smoker)

le.fit(df.region.drop_duplicates())
df.region = le.transform(df.region)

# Splitting the dataset into X and y (features and target variable)
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
# apply feature scaling to reduce the range of the data (excluding the extreme values)
sc_X = StandardScaler()
X = sc_X.fit_transform(X)

# train_test_split function
def train_test_split(data,labels,testRatio=0.3):
    n=len(data)
    print("train_test_split function called with data size: ",n," and testRatio: ",testRatio)
    testSize=int(n*testRatio) 
    print("testSize: ",testSize)
    print("trainSize: ",n-testSize)
    testIndices=np.random.choice(n,testSize,replace=False)
    trainIndices=[i for i in range(n) if i not in testIndices]
    return data[trainIndices],data[testIndices],labels[trainIndices],labels[testIndices]

# Splitting the dataset into the Training set and Test set 
train_Size = 1000 #int(input("Enter the train size: "))
testRatio = 1- (train_Size / len(X))


X_train,X_test,y_train,y_test = train_test_split(X,y, testRatio)

# target values scaling (for SVR model)
y_reshaped = y_train.reshape(len(y_train),1)
sc_y = StandardScaler()
y_new = sc_y.fit_transform(y_reshaped)


# Model 1: Linear Regression
LinearRegressor = LinearRegression()
LinearRegressor.fit(X_train, y_train)


# Predicting the Test set results
y_pred_LR = LinearRegressor.predict(X_test)

# calculate the mean squared error  
mseRegression = mean_squared_error(y_test, y_pred_LR)
r2_scoreRegression = r2_score(y_test, y_pred_LR)

# Model 2: SVR (linear kernel)
SVR_linearRegressor = SVR(kernel='linear')
SVR_linearRegressor.fit(X_train, y_new.ravel())


#y_pred = SVR_linearRegressor.predict(X_test)
y_pred_SVR = sc_y.inverse_transform(SVR_linearRegressor.predict(X_test).reshape(-1,1))

# calculate the mean squared error
r2_scoreSVM = r2_score(y_test, y_pred_SVR)
mseSVM = mean_squared_error(y_test, y_pred_SVR)

# Model 3: SVR (rbf kernel)
SVR_RBF_Regressor = SVR(kernel='rbf')
SVR_RBF_Regressor.fit(X_train, y_new.ravel())


#y_pred = SVR_linearRegressor.predict(X_test)
y_pred_SVR_RBF = sc_y.inverse_transform(SVR_RBF_Regressor.predict(X_test).reshape(-1,1))

# calculate the mean squared error
r2_scoreSVM_RBF = r2_score(y_test, y_pred_SVR_RBF)
mseSVM_RBF = mean_squared_error(y_test, y_pred_SVR_RBF)

# Model 4: Random Forest Regressor
forest_regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
forest_regressor.fit(X_train, y_train)

y_pred_forest = forest_regressor.predict(X_test)
r2_scoreForest = r2_score(y_test, y_pred_forest)
mseForest = mean_squared_error(y_test, y_pred_forest)

# Model 5: KNN Regressor
knn_regressor = KNeighborsRegressor(n_neighbors = 5)
knn_regressor.fit(X_train, y_train)

y_pred_knn = knn_regressor.predict(X_test)
r2_scoreKNN = r2_score(y_test, y_pred_knn)
mseKNN = mean_squared_error(y_test, y_pred_knn)


# print the results
print("Linear Regression: Mean Squared Error: ",mseRegression)
print("SVR_linear Mean Squared Error: ",mseSVM)
print("SVR_RBF Mean Squared Error: ",mseSVM_RBF)
print("Random Forest: Mean Squared Error: ",mseForest)
print("KNN: Mean Squared Error: ",mseKNN)

print("\n###########################################\n")


print("Linear Regression: R2 Score: ",r2_scoreRegression)
print("SVR_linear R2 Score: ",r2_scoreSVM)
print("SVR_RBF R2 Score: ",r2_scoreSVM_RBF)
print("Random Forest: R2 Score: ",r2_scoreForest)
print("KNN: R2 Score: ",r2_scoreKNN)

# print the most accurate model
models = {"Linear Regression": mseRegression, "SVR Linear": mseSVM, "SVR RBF": mseSVM_RBF, "Random Forest": mseForest, "KNN": mseKNN}
bestModel = min(models, key=models.get)
print("\n###########################################\n")
print("The best model is: ", bestModel)
print("\n###########################################\n")

