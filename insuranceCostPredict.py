# insurance.csv dataset contains the following columns:
# age: integer
# sex: (male, female)
# bmi: float
# children: integer
# smoker: (yes, no)
# region: (northeast, southeast, southwest, northwest)
# charges: float

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder 
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# Importing the dataset
dataset = pd.read_csv('insurance.csv')



# convert data set to a pandas dataframe
df = pd.DataFrame(dataset)


# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()

df = df.apply(labelencoder.fit_transform)

X = df.iloc[:, :-1].values
y = df.iloc[:, 6].values

def train_test_split(data,labels,testRatio=0.3):
    n=len(data)
    print("train_test_split function called with data size: ",n," and testRatio: ",testRatio)
    testSize=int(n*testRatio) 
    print("testSize: ",testSize)
    print("trainSize: ",n-testSize)
    testIndices=np.random.choice(n,testSize,replace=False)
    trainIndices=[i for i in range(n) if i not in testIndices]
    return data[trainIndices],data[testIndices],labels[trainIndices],labels[testIndices]


train_Size = 1000 #int(input("Enter the train size: "))
testRatio = 1- (train_Size / len(X))


X_train,X_test,y_train,y_test =train_test_split(X,y, testRatio)




# Fitting Multiple Linear Regression to the Training set
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

# calculate the mean squared error  
mseRegression = mean_squared_error(y_test, y_pred)
r2_scoreRegression = r2_score(y_test, y_pred)
print("R2 Score: ",r2_scoreRegression)
print("Mean Squared Error: ",mseRegression)

# Fitting Multiple Linear SVM to the Training set
from sklearn.svm import SVR
regressor = SVR(kernel = 'linear')
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

# calculate the mean squared error
r2_scoreSVM = r2_score(y_test, y_pred)
print("R2 Score: ",r2_scoreSVM)
mseSVM = mean_squared_error(y_test, y_pred)
print("Mean Squared Error: ",mseSVM)



# Fitting Naive Bayes to the Training set
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# calculate the mean squared error
r2_scoreNaiveBayes = r2_score(y_test, y_pred)
print("R2 Score: ",r2_scoreNaiveBayes)
mseNaiveBayes = mean_squared_error(y_test, y_pred)
print("Mean Squared Error: ",mseNaiveBayes)


# Comparing the results

print("Mean Squared Error for Regression: ",mseRegression)
print("Mean Squared Error for SVM: ",mseSVM)
print("Mean Squared Error for Naive Bayes: ",mseNaiveBayes)

if mseRegression < mseSVM and mseRegression < mseNaiveBayes:
    print("Regression is the best model")
elif mseSVM < mseRegression and mseSVM < mseNaiveBayes:
    print("SVM is the best model")
else:
    print("Naive Bayes is the best model")