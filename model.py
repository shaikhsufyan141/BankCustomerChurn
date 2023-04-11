import pandas as pd
import numpy as np
import pickle
import warnings
import joblib
warnings.filterwarnings("ignore")
from sklearn.ensemble import GradientBoostingClassifier
data=pd.read_csv(r"C:\Users\shaik\Downloads\Churn_Modelling.csv",index_col=0)
data.head()
data.Exited.value_counts()


data.shape


data.drop("Surname",axis=1,inplace=True)
data.drop("CustomerId",axis=1,inplace=True)
data.drop("Tenure",axis=1,inplace=True)

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

for i in data.columns:
    if data[i].dtype=="bool" or data[i].dtype=="object":
        data[i] = le.fit_transform(data[i])

data.head()

X = data.values[:,:-1]
Y = data.values[:,-1]

print(X.shape)
print(Y.shape)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(X)
X = scaler.transform(X)
print(X)

Y = Y.astype(int)

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=10)
print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)

# SMOTING 
print("Before OverSampling, counts of label '1': ", (sum(Y_train == 1)))
print("Before OverSampling, counts of label '0': ", (sum(Y_train == 0)))
  
# import SMOTE from imblearn library
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state =10,k_neighbors=5)
X_train_res, Y_train_res = sm.fit_resample(X_train, Y_train)
  
print('After OverSampling, the shape of train_X: ', (X_train_res.shape))
print('After OverSampling, the shape of train_y: ', (Y_train_res.shape))
  
print("After OverSampling, counts of label '1': ", (sum(Y_train_res == 1)))
print("After OverSampling, counts of label '0': ", (sum(Y_train_res == 0)))

#create a model
Gb=GradientBoostingClassifier(n_estimators=50,random_state=10)

#fitting the training data into the model
Gb.fit(X_train_res,Y_train_res)
Y_pred=Gb.predict(X_test)
#print(list(zip(Y_test, Y_pred)))

#print(list(zip(adult_df_rev.columns[:-1],classifier.coef_.ravel())))
#print(Gb.intercept_)

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

cm=confusion_matrix(Y_test,Y_pred)
print("Confusion matrix by Gradient Boosting Classifier:\n",cm)
class_r=classification_report(Y_test,Y_pred)
print("Classfication rep by Gradient Boosting Classifier:")
print(class_r)
acc_s=accuracy_score(Y_test,Y_pred)
print("Accuracy score by  Gradient Boosting Classifier:",acc_s)


filename = r'classifier.sav'
# dump the sclaer in pwd

joblib.dump(scaler,'gb_scaler.pkl')
pickle.dump(Gb, open(filename,'wb'))
loaded_model = pickle.load(open(filename,'rb'))
pickle.dump(Gb,open("churn.pkl",'wb'))
model = pickle.load(open("churn.pkl",'rb'))
Gb_scaler = joblib.load('gb_scaler.pkl')
