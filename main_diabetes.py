import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

  

def diabetes():
  ds = pd.read_csv('C:/Users/IRPT/Desktop/Diabetes App/Diabetes Death Dataset.csv') 
  X = ds.drop(['Diabetes'],axis=1)
  Y = ds['Diabetes']

  sc = StandardScaler()
  sc.fit(X)
  std_data = sc.transform(X)
  X = std_data
  Y = ds['Diabetes']

  X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, stratify=Y, random_state=2)

  classifier = svm.SVC(kernel='linear')
  classifier.fit(X_train, Y_train)

  X_train_prediction = classifier.predict(X_train)
  training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
  X_test_prediction = classifier.predict(X_test)
  test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

  a=int(input("Age:\t"))
  b=int(input("Anaemia:\t"))
  c=int(input("Creatinine:\t"))
  d=int(input("Sex:\t"))
  e=int(input("Ejection Fraction:\t"))
  f=int(input("High Blood Pressure:\t"))
  g=int(input("Platelets:\t"))
  h=float(input("Serum Creatinine:\t"))
  i=int(input("Serum Sodium:\t"))
  j=int(input("Smoking:\t"))
  k=int(input("SmokesperDay:\t"))
  input_data=tuple([a,b,c,d,e,f,g,h,i,j,k])
  input_data_as_numpy_array = np.asarray(input_data)
  input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
  std_data = sc.transform(input_data_reshaped)
  #print(std_data)
  prediction = classifier.predict(std_data)
  #print(prediction)
  if (prediction[0] == 0):
    print('The person is not diabetic')
  else:
    print('The person is diabetic')

diabetes()