from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, confusion_matrix
import json
import pickle as pk
import train as t
import data_prep as dp


def predTrain(model, scale = True):
    X_train = dp.X_train.copy() # prevent accidental overwrite
    if scale: # always scales except when told not to.
        scaler = t.loadScaler() # resused from train module
        X_train[['Time', 'Amount']] = scaler.transform(X_train[['Time', 'Amount']]) # transform with dp's fitted scaler
    y_pred = model.predict(X_train)
    return y_pred

def predTest(model, scale = True):
    X_test = dp.X_test.copy() # prevent accidental overwrite
    if scale:
        scaler = t.loadScaler() # resused from train module
        X_test[['Time', 'Amount']] = scaler.transform(X_test[['Time', 'Amount']]) # transform with dp's fitted scaler
    y_pred = model.predict(X_test)
    return y_pred


def createReport(model):
    name = type(model).__name__
    MODELSTAT = {}

    print(f"Predicting test results for model {name}")
    print(f"Predicting train results for model {name}")

    y_pred = predTest()
    y_pred = predTrain()

    cm = confusion_matrix(dp.y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()   # works for binary only
    MODELSTAT[name] = {
        "accuracy":          round(accuracy_score(dp.y_test, y_pred),4),
        "precision":         round(precision_score(dp.y_test, y_pred),4),
        "recall":            round(recall_score(dp.y_test, y_pred),4),
        "f1":                round(f1_score(dp.y_test, y_pred),4),
        "confusion matrix": {
                            "TN": int(tn),
                            "FP": int(fp),
                            "FN": int(fn), 
                            "TP": int(tp),
        }
}
    saveReport(name, MODELSTAT)

#============================================

def saveReport(name, report):
    with open(f"results{name}.json", "a") as f:
        json.dump(report, f, indent=2)

#============================================

models = {
    "logistic_regression": t.trainModel(LogisticRegression()),
    "knn": t.trainModel(KNeighborsClassifier()), #default is n_neighbors = 5
    "decision_tree": t.trainModel(DecisionTreeClassifier(max_depth=10, random_state=42)),
}

#can be main
for name, model in models.items():
    createReport(model)
t.saveModel(models['knn'])