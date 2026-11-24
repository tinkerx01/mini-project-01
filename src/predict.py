from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, confusion_matrix
import data_prep as dp
import json


def _evaluate(model, X, y, scale=True):
    # shared core function
    if scale:
        X = dp.scaleX(X) #breaks if no scaler pkl
    y_pred = model.predict(X)

    cm = confusion_matrix(y, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return {
        "accuracy":  round(accuracy_score(y, y_pred), 4),
        "precision": round(precision_score(y, y_pred), 4),
        "recall":    round(recall_score(y, y_pred), 4),
        "f1":        round(f1_score(y, y_pred), 4),
        "confusion_matrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
    }

# ============================================================================ #

def reportTrain(model, scale=True):
    name = type(model).__name__
    model = dp.loadModel(name)   # breaks if no model pkl
    print(f"[TRAIN] evaluating {name}")
    stats = _evaluate(model, dp.X_train, dp.y_train, scale=scale)
    _save(f"results_train_{name}.json", {name: stats})
    return stats

# ============================================================================ #

def reportTest(model, scale=True):
    name = type(model).__name__
    model = dp.loadModel(name)  
    print(f"[TEST] evaluating {name}")
    stats = _evaluate(model, dp.X_test, dp.y_test, scale=scale)
    _save(f"results_test_{name}.json", {name: stats})
    return stats

# ============================================================================ #

def _save(filename, payload):
    path = dp.PROJECT_ROOT / filename
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {path}")

