from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, confusion_matrix
import train as t
import data_prep as dp
import json

models = {
    "logistic_regression": t.model_LR,
    "knn": t.model_knn,
    "decision_tree": t.model_dt,
}

MODELS = {}
for name, model in models.items():
    y_pred = model.predict(dp.X_test)
    cm = confusion_matrix(dp.y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()   # works for binary only
    MODELS[name] = {
        "accuracy":          accuracy_score(dp.y_test, y_pred),
        "precision":         precision_score(dp.y_test, y_pred),
        "recall":            recall_score(dp.y_test, y_pred),
        "f1":                f1_score(dp.y_test, y_pred),
        "confusion matrix": {
                            "TN": int(tn),
                            "FP": int(fp),
                            "FN": int(fn), 
                            "TP": int(tp),
        }
}


with open("results.json", "w") as f:
    json.dump(MODELS, f, indent=2)
