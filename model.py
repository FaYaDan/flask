import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
import pickle


dataset = pd.read_csv('final_test.csv')

# Menampilkan dataset
print(dataset.head())

"""Mengecek Missing Value"""

missing_values = dataset.isnull().sum()
print(missing_values)

"""Menghilangkan Missing Value"""

data = dataset.dropna()

"""Mengecek Type Data"""

data.dtypes

"""Mengubah tipe data Age dan Height Menjadi Integer"""

data['height'] = data['height'].astype(int)
data['age'] = data['age'].astype(int)

"""Mengecek jumlah data dalam masing masing kelas"""

data['size'].value_counts()

"""Membagi Data Menjadi Training dan Testing"""
# Memisahkan fitur (x) dan target (y)
x = data[['age', 'height', 'weight']]
y = data['size']

# Membagi data menjadi set pelatihan dan pengujian
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20,random_state=42)

print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)

"""Klasifikasi Menggunakan Decision tree"""

# Model training
model = DecisionTreeClassifier()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

# Menghitung akurasi model
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)

print(classification_report(y_test, y_pred))
print(y_test)

#simpan data h5? #
# Simpan model ke file
with open('model_decision_tree.pkl', 'wb') as file:
    pickle.dump(model, file)



#===================#
#cobain pake data baru
#d = {'weight': [50], 'age': [20], 'height': [180]}
##Me = pd.DataFrame(data=d)

#Me = Me.reindex(columns=['age', 'height', 'weight'])

# Melakukan prediksi menggunakan model
#prediction = model.predict(Me)[0]
#print("Predicted Size:", prediction)
