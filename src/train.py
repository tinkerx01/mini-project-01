from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import data_prep as dp
import pickle

MODELS_DIR = dp.PROJECT_ROOT / "models"

def trainModel(model, scale = True):
    print(f"training model {dp.modelName(model)}")
    X_train = dp.X_train.copy() # saving so we don't transform the actual X_train in data_pred. (reusability)
    if scale:
        scaler = dp.loadScaler()
        X_train[['Time', 'Amount']]= scaler.transform(X_train[['Time', 'Amount']])
    model.fit(X_train, dp.y_train) # fits model after/without scaling the training data
    return model

# ============================================================================ #

def saveModel(model, filename=None, overwrite=False):
    if filename is None:
        filename = f"{dp.modelName(model)}.pkl"
    path = MODELS_DIR / filename
    if path.exists() and not overwrite:
        print(f"Model file {path} already exists; pass overwrite=True to replace.")
        return
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"Saved model to {path}")
