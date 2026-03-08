#-------------------------------------------------------------------------
# AUTHOR: your name
# FILENAME: knn.py
# SPECIFICATION: Compute the LOO-CV error rate for a 1NN classifier on
#                the spam/ham email classification task using 20 word-count features.
# FOR: CS 4210- Assignment #2
# TIME SPENT: N/A
#-----------------------------------------------------------*/

#IMPORTANT NOTE: YOU ARE ALLOWED TO USE ANY PYTHON LIBRARY TO COMPLETE THIS PROGRAM

#Importing some Python libraries
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

db = []
df = pd.read_csv('email_classification.csv')
for _, row in df.iterrows():
    db.append(row.tolist())

errors = 0

for i in range(len(db)):
    X = [list(map(float, db[j][:-1])) for j in range(len(db)) if j != i]
    Y = [0.0 if db[j][-1] == 'ham' else 1.0 for j in range(len(db)) if j != i]

    testSample = list(map(float, db[i][:-1]))

    clf = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
    clf.fit(X, Y)

    class_predicted = clf.predict([testSample])[0]

    true_label = 0.0 if db[i][-1] == 'ham' else 1.0
    if class_predicted != true_label:
        errors += 1

error_rate = errors / len(db)
print(f"Error Rate: {error_rate:.2f}")
