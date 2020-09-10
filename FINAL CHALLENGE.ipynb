#Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, accuracy_score
from sklearn.linear_model import LogisticRegression
from pandas import Series, DataFrame

train = pd.read_csv(r"C:/Users/MICHAEL/Desktop/mbark-sample-superstore/Train (3).csv")
test = pd.read_csv(r"C:/Users/MICHAEL/Desktop/mbark-sample-superstore/Test (3).csv")
X = np.array(train.drop(['Applicant_ID', 'default_status'], axis=1))
y = np.array(train["default_status"])
test_data = np.array(test.drop(['Applicant_ID'], axis=1))


Mike = train.isnull().sum()
Mike2 = test.isnull().sum()
print(Mike, Mike2)

#filling misssing values
#Replacing categorical null values
#X['form_field47'].value_counts()
#X['form_field47'] = X['form_field47'].fillna(value='None')

#imputer = SimpleImputer(missing_values=np.nan, fill_value='None', strategy='most_frequent')
#imputer = imputer.fit(X[['form_fiel']])
#X['form_field47']= imputer.transform(X[['form_field47']]).ravel()
#X.isnull().sum()



imputer = SimpleImputer(missing_values=np.nan, fill_value=None, strategy="most_frequent")
imputer = imputer.fit(X[:, :46])
X[:, :46]= imputer.transform(X[:, :46])
imputer = SimpleImputer(missing_values=np.nan, fill_value=None, strategy="most_frequent")
imputer = imputer.fit(test_data[:, :46])
test_data[:, :46]= imputer.transform(test_data[:, :46])


imputer = SimpleImputer(missing_values=np.nan, fill_value=0.0, strategy="mean")
imputer = imputer.fit(X[:, [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,47,48,49]])
X[:, [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,47,48,49]]= imputer.transform(X[:, [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,47,48,49]])
imputer = SimpleImputer(missing_values=np.nan, fill_value=0.0, strategy="mean")
imputer = imputer.fit(test_data[:, [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,47,48,49]])
test_data[:, [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,47,48,49]]= imputer.transform(test_data[:, [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,47,48,49]])



#Changing our Dependent variable y from categorical to numerical data using Label Encoder
Label = LabelEncoder()
y = Label.fit_transform(y)

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer(transformers=[('One_hot_encoder', OneHotEncoder(categories='auto'),[46])],remainder='passthrough')
X = np.array(ct.fit_transform(X), dtype=np.object)
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer(transformers=[('One_hot_encoder', OneHotEncoder(categories='auto'),[46])],remainder='passthrough')
test_data = np.array(ct.fit_transform(test_data), dtype=np.object)


X = X[:, 1:]
test_data= test_data[:, 1:]

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=0)

pipe = Pipeline([("scaler", StandardScaler()), ("model", RandomForestClassifier())])
pipe2 = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression())])
pipe.fit(X_train, y_train)
pipe2.fit(X_train, y_train)
prediction = pipe.predict(X_test)
prediction2 = pipe2.predict(X_test)
score = accuracy_score(y_test, prediction)
score2 = accuracy_score(y_test, prediction2)
score3 = log_loss(y_test, prediction)

pred = pipe.predict(test_data)

Dataset_test123 = pd.read_csv("C:/Users/MICHAEL/Desktop/New folder/Test.csv")
df = DataFrame(test)
df= df.iloc[:, 0]
df = DataFrame(df)
df = df[['Applicant_id', 'Churn']]
df1 = DataFrame(pred)
df['Churn'] = pred

df.to_csv(r'C:/Users/MICHAEL/Desktop/New folder\sumbission123.csv', index=None)

DataFrame({"Applicant_ID": test["Applicant_ID"], "default_status":pred}).to_csv(r"C:/Users/MICHAEL/Desktop/mbark-sample-superstore/submission_file1.csv", index=False)

