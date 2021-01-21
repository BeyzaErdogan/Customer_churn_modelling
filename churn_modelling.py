import pandas as pd
import numpy as np

veriler = pd.read_csv("Churn_Modelling.csv")

#Veri seti incelendi ve ilk 3 sütun isim ve indeks sütunları bunlar müşterinin kalıp kalmayacağını belirlemez. Model ezber yapabilir. Bu yüzden kaldırılmalıdır.
#Yapay sinir ağları 0 ile 1 arasında değer alır. Bu yüzden encoding ve standartscaler yapmaya ihtiyaç var.
X = veriler.iloc[:,3:13].values
Y = veriler.iloc[:,13].values

from sklearn import preprocessing

le = preprocessing.LabelEncoder()

X[:,1] = le.fit_transform(X[:,1])

le2 = preprocessing.LabelEncoder()

X[:,2] = le2.fit_transform(X[:,2])

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

ohe = ColumnTransformer([("ohe",OneHotEncoder(dtype=float),[1])],remainder="passthrough") #veri tipi float, [1]-> 1.kolon

X = ohe.fit_transform(X)
X = X[:,1:]

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=0)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)


import tensorflow
import keras
from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential() #bir yapay sinir ağı oluşturur.
#iki gizli katman ve bir çıkış katmanı oluşturulur.
classifier.add(Dense(6, kernel_initializer='random_uniform', activation = "relu", input_dim = 11))#ara katman 6 neron olarak karar verdik. 11 giriş, ara katman 6 nöron, ve 1 çıkış aslında üçgen yapısı oluşturmak gibi.
classifier.add(Dense(6, kernel_initializer='random_uniform', activation = "relu"))
#tavsiye: gizli ve giriş katmanda linear, çıkış sigmoid fonk kullanılması öneriliyor.
classifier.add(Dense(1, kernel_initializer='random_uniform', activation = "sigmoid"))#output katmanı
#sınıf etiketi binary değerler(1 ve 0) bu yüzden loss fonk olarak binary_crossentropy
classifier.compile(optimizer = 'adam',loss="binary_crossentropy",metrics=['accuracy']) #ağırlıklar nasıl optimize edilecek, adam => stokastik gradient descent yöntemi(sinapsisleri optimize eder) 
classifier.fit(X_train, y_train, epochs=50)
y_pred = classifier.predict(X_test)

y_pred = (y_pred > 0.5)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)


