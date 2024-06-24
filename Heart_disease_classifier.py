# %%
import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as  plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report


# %%
df =  pd.read_csv('framingham.csv')
df.head()

# %%
df.isna().sum()

# %%
cols_with_missing_values_above_tf = df.columns
for col in cols_with_missing_values_above_tf:
    df[col]=df[col].fillna(df[col].mode()[0])
df.isna().sum()

# %%
features = df.drop('TenYearCHD',axis=1).values
labels = df['TenYearCHD'].values
X_train,X_test,Y_train,Y_test = train_test_split(features,labels,random_state=42,test_size=0.3)
#models = [RandomForestClassifier(),]
results ={'RandomForestClassifier':RandomForestClassifier(random_state=42,n_estimators=2,),'LogisticRegression':LogisticRegression(max_iter=10000,tol=0.0001)}
for key ,value in results.items():
    a = value
    a.fit(X_train,Y_train)
    y_predict = a.predict(X_test)
    print(key,"\n")
    print(confusion_matrix(Y_test,y_predict),"\n")
    print(classification_report(Y_test,y_predict),"\n")
    print("The model score on test data:",a.score(X_test,Y_test),"\n\n")
    results[key]= a.score(X_train,Y_train)
print(results)
    


# %%


# %%



