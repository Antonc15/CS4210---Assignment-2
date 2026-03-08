#-------------------------------------------------------------------------
# AUTHOR: Anton Clark
# FILENAME: naive_bayes.py
# SPECIFICATION: Read weather training data, train a Gaussian Naive Bayes classifier,
#                then predict test instances and print those with confidence >= 0.75
# FOR: CS 4210- Assignment #2
# TIME SPENT: <30 mins
#-----------------------------------------------------------*/

#IMPORTANT NOTE: YOU ARE ALLOWED TO USE ANY PYTHON LIBRARY TO COMPLETE THIS PROGRAM

#Importing some Python libraries
from sklearn.naive_bayes import GaussianNB
import csv

dbTraining = []
dbTest = []

outlook_map     = {'Sunny': 1, 'Overcast': 2, 'Rain': 3}
temperature_map = {'Hot': 1, 'Mild': 2, 'Cool': 3}
humidity_map    = {'High': 1, 'Normal': 2}
wind_map        = {'Weak': 1, 'Strong': 2}
label_map       = {'Yes': 1, 'No': 2}
label_reverse   = {1: 'Yes', 2: 'No'}

with open('weather_training.csv', 'r') as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        dbTraining.append(row)

X = []
for row in dbTraining:
    X.append([outlook_map[row[1]], temperature_map[row[2]], humidity_map[row[3]], wind_map[row[4]]])

Y = []
for row in dbTraining:
    Y.append(label_map[row[5]])

clf = GaussianNB(var_smoothing=1e-9)
clf.fit(X, Y)

with open('weather_test.csv', 'r') as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        dbTest.append(row)

print(f"{'Day':<8} {'Outlook':<10} {'Temperature':<13} {'Humidity':<10} {'Wind':<8} {'PlayTennis':<12} {'Confidence'}")

for data in dbTest:
    features = [outlook_map[data[1]], temperature_map[data[2]], humidity_map[data[3]], wind_map[data[4]]]
    probs = clf.predict_proba([features])[0]

    confidence = max(probs)
    predicted_label = label_reverse[clf.classes_[probs.argmax()]]

    if confidence >= 0.75:
        print(f"{data[0]:<8} {data[1]:<10} {data[2]:<13} {data[3]:<10} {data[4]:<8} {predicted_label:<12} {confidence:.2f}")
