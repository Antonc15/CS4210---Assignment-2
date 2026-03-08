#-------------------------------------------------------------------------
# AUTHOR: Anton Clark
# FILENAME: decision_tree_2.py
# SPECIFICATION: Train and test decision trees using 3 training sets of different sizes,
#                repeated 10 times each, reporting average accuracy per model.
# FOR: CS 4210- Assignment #2
# TIME SPENT: 50 mins
#-----------------------------------------------------------*/

#IMPORTANT NOTE: YOU ARE ALLOWED TO USE ANY PYTHON LIBRARY TO COMPLETE THIS PROGRAM

#Importing some Python libraries
from sklearn import tree
import csv

dataSets = ['contact_lens_training_1.csv', 'contact_lens_training_2.csv', 'contact_lens_training_3.csv']

age_map          = {'Young': 1, 'Prepresbyopic': 2, 'Presbyopic': 3}
prescription_map = {'Myope': 1, 'Hypermetrope': 2}
astigmatism_map  = {'Yes': 1, 'No': 2}
tear_map         = {'Normal': 1, 'Reduced': 2}
label_map        = {'Yes': 1, 'No': 2}

def encode(row):
    return [
        age_map[row[0]],
        prescription_map[row[1]],
        astigmatism_map[row[2]],
        tear_map[row[3]]
    ]

dbTest = []

with open('contact_lens_test.csv', 'r') as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        dbTest.append(row)

for ds in dataSets:

    dbTraining = []
    X = []
    Y = []

    with open(ds, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            dbTraining.append(row)

    for row in dbTraining:
        X.append(encode(row))

    for row in dbTraining:
        Y.append(label_map[row[4]])

    total_accuracy = 0.0

    # x10
    for i in range(10):
        clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=5)
        clf = clf.fit(X, Y)

        correct = 0

        for data in dbTest:
            test_features = encode(data)
            class_predicted = clf.predict([test_features])[0]
            true_label = label_map[data[4]]

            if class_predicted == true_label:
                correct += 1

        run_accuracy = correct / len(dbTest)
        total_accuracy += run_accuracy

    # Avg
    avg_accuracy = total_accuracy / 10

    print(f"Accuracy when training on {ds}: {avg_accuracy:.1f}")
