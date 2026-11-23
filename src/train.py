from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import data_prep as dp
import pickle

MODELS_DIR = dp.PROJECT_ROOT / "models"

def trainModel(model):
    print(f"training model {type(model).__name__}")
    X_train = dp.X_train.copy()
    if not isinstance(model, DecisionTreeClassifier):
        X_train[['Time', 'Amount']]= dp.scaler.transform(X_train[['Time', 'Amount']])
    model.fit(X_train, dp.y_train)
    return model

def saveModel(model, filename=None):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if filename is None:
        filename = f"{type(model).__name__}.pkl"
    path = MODELS_DIR / filename
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"Saved model to {path}")

