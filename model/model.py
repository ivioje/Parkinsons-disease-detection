from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import accuracy_score
import pandas as pd
import pickle

parkinsons_data = pd.read_pickle('/data/processed_data.pkl')
save_data = '/model/svm.pkl'

Y = parkinsons_data['status']
X = parkinsons_data.drop(columns =['name', 'status'], axis = 1)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test)

model = svm.SVC(kernel='linear')

# training the svm model with training data
model.fit (X_train, Y_train)

# accuracy score on training data
X_train_prediction = model.predict (X_train)
training_data_accuracy = accuracy_score (Y_train, X_train_prediction)

print('Accuracy score of training data:', training_data_accuracy)

X_test_prediction = model.predict (X_test)
test_data_accuracy = accuracy_score (Y_test, X_test_prediction)

print('Accuracy score of test data:', test_data_accuracy)

with open(save_data, "wb") as pickle_out:
    pickle.dump(model, pickle_out)