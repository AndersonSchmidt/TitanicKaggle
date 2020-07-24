# Titanic Machine Learning

import numpy as np
import pandas as pd

#Importing the dataset
train = pd.read_csv('train.csv')
test  = pd.read_csv('test.csv')
X_train = train.iloc[:, [2,4,5,6,7,9,11]].values
y_train = train.iloc[:, 1].values
X_test = test.iloc[:, [1,3,4,5,6,8,10]].values
df_X_train=pd.DataFrame(X_train) #DataFrame just for visualizing the data
df_X_test=pd.DataFrame(X_test) #DataFrame just for visualizing the data

# Taking care of missing data
# Numeric Data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean', verbose=0)

imputer = imputer.fit(X_train[:, 2:3])
X_train[:, 2:3] = imputer.transform(X_train[:, 2:3])

imputer = imputer.fit(X_test[:, 2:3])
X_test[:, 2:3] = imputer.transform(X_test[:, 2:3])

imputer = imputer.fit(X_test[:, 5:6])
X_test[:, 5:6] = imputer.transform(X_test[:, 5:6])

# String Data
imputer = SimpleImputer(missing_values = np.nan, strategy = 'most_frequent', verbose=0)

imputer = imputer.fit(X_train[:, 6:7])
X_train[:, 6:7] = imputer.transform(X_train[:, 6:7])


# Encoding Categorical Data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X_train[:, 1] = labelencoder_X.fit_transform(X_train[:,1])
onehotencoder = OneHotEncoder(categorical_features = [1])
X_train = onehotencoder.fit_transform(X_train).toarray()

X_test[:, 1] = labelencoder_X.fit_transform(X_test[:,1])
onehotencoder = OneHotEncoder(categorical_features = [1])
X_test = onehotencoder.fit_transform(X_test).toarray()

# Avoiding the Dummy Variable Trap
X_train = X_train[:, 1:]
X_test = X_test[:, 1:]


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

# Fitting the classifier to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

aa = np.isnan(X_test)



