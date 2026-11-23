from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, confusion_matrix
import train as t
import data_prep as dp
import json


def createReport(model):
    name = type(model).__name__
    MODELSTAT = {}

    X_test = dp.X_test.copy()
    if not isinstance(model, DecisionTreeClassifier):
        X_test[['Time', 'Amount']]= dp.scaler.transform(X_test[['Time', 'Amount']])

    print(f"Predicting results for model {name}")
    y_pred = model.predict(X_test)
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