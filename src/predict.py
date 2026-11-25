from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
import data_prep as dp
import json
import numpy as np


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
    name = dp.modelName(model)
    model = dp.loadModel(name)   # breaks if no model pkl
    print(f"[TRAIN] evaluating {name}")
    stats = _evaluate(model, dp.X_train, dp.y_train, scale=scale)
    _save(f"results_train_{name}.json", {name: stats})
    return stats

# ============================================================================ #

def reportTest(model, scale=True):
    name = dp.modelName(model)
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

# ============================================================================ #

def _aggregate(per_fold):
    metrics = ("accuracy", "precision", "recall", "f1")
    return {
        m: {
            "mean": round(float(np.mean([f[m] for f in per_fold])), 4),
            "std":  round(float(np.std([f[m] for f in per_fold])), 4),
        }
        for m in metrics
    }

# ============================================================================ #

def crossValidate(model, folds=5, scale=True, save_per_fold=True, save_aggregate=True):
    name = dp.modelName(model)
    model = dp.loadModel(name)

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    per_fold = []

    for i, (tr_idx, va_idx) in enumerate(skf.split(dp.X_train, dp.y_train)):
        X_tr = dp.X_train.iloc[tr_idx].copy()
        X_va = dp.X_train.iloc[va_idx].copy()
        y_tr = dp.y_train.iloc[tr_idx]
        y_va = dp.y_train.iloc[va_idx]

        if scale:
            _, X_tr, X_va = dp.fitScaleX(X_tr, X_va)

        fold_model = clone(model).fit(X_tr, y_tr)
        fold_stats = _evaluate(fold_model, X_va, y_va, scale=False)
        per_fold.append(fold_stats)

        if save_per_fold:
            _save(f"results_cv_{name}_fold{i}.json", {name: fold_stats})

    aggregate = _aggregate(per_fold)
    if save_aggregate:
        _save(f"results_cv_{name}.json", {name: aggregate})

    return {"per_fold": per_fold, "aggregate": aggregate}



