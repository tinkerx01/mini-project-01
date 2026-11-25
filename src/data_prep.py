import os
import shutil
import pandas as pd
import kagglehub
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import pickle


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CSV_PATH = DATA_DIR / "creditcard.csv"
MODELS_DIR = PROJECT_ROOT / "models"


def is_valid_csv(path):
    """Return True if file exists and is a readable, non-empty CSV."""
    if not os.path.isfile(path):
        return False
    if os.path.getsize(path) < 1024:  # sanity: too small to be real
        return False
    try:
        # Peek at the file to ensure it's parseable
        pd.read_csv(path, nrows=5)
        return True
    except Exception:
        return False

#========================================================================

def loader():
    # handling existing dataset
    if is_valid_csv(CSV_PATH):
        print(f"Found valid file at {CSV_PATH}, skipping download.")
    else:
        print(f"{CSV_PATH} missing or corrupted, downloading...")
        os.makedirs(DATA_DIR, exist_ok=True)

        dataset_path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
        src = os.path.join(dataset_path, "creditcard.csv")

        shutil.copy(src, CSV_PATH)
        print(f"Saved to {CSV_PATH}")
    # creating df
    df =  pd.read_csv(CSV_PATH) 
    return df
    

#========================================================================

def statistics():
    print(f"Number of Samples: {len(df)}")
    print(f"Number of features: {df.shape[1]-1}") # '-1' is the 'Class' column

#========================================================================

def distribution(): # incase someone's curious
    labels = {0: "Normal", 1: "Fraud"}
    count = df["Class"].value_counts()
    for cls, n in count.items():
        print(f"{labels[cls]} (Class {cls}): {n}  ({n/len(df)*100:.3f}%)")

#========================================================================

# dataset missing checks
def MissingValues():
    print(f"inf Values: {np.isinf(df.select_dtypes(include=np.number)).sum().sum()}") #infs
    print(f"null Values: {df.isnull().sum().sum()}") # nulls
    print(f"Min Time: {df['Time'].min()}") # negative elapsed time
    print(f"Min Amount: {df['Amount'].min()}") # negative Amount 
    print(f"Is time column acsending: {df['Time'].is_monotonic_increasing}") # time column is ascending? (could've read the docs but just incase)
    print(df.drop(columns=['Time', 'Amount', 'Class']).mean().describe()) # col values distribution (again could've read the docs)

#========================================================================

def duplicates():
    print(f"Number of duplicated samples: {df.duplicated().sum()}")
    dupes = df[df.duplicated(keep=False)].sort_values(list(df.columns))
    print(dupes['Class'].value_counts())


# ============================================================================ #

def fitScaler(overwrite=False):
    path = MODELS_DIR / "scaler.pkl"
    if path.exists() and not overwrite:
        print(f"Scaler already exists at {path}; pass overwrite=True to refit.")
        return
    scaler = StandardScaler().fit(X_train[['Time', 'Amount']])
    saveScaler(scaler, overwrite=True)
    print(f"Fitted and saved scaler to {path}")

# ============================================================================ #

def saveScaler(scaler, path=None, overwrite=False):
    # if scaler found, don't touch 
    path = path or (MODELS_DIR / "scaler.pkl")
    if path.exists() and not overwrite:
        print(f"scaler file already exists; pass overwrite=True to replace.")
        return
    # if not found, save the new one.
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(scaler, f)
        print("saved scaler.")

# ============================================================================ #

def loadScaler(path=None):
    path = path or (MODELS_DIR / "scaler.pkl")
    if not path.exists():
        raise FileNotFoundError(f"Scaler doesn't exist. Aborting...")
    print(f"Loading scaler from {path}")
    with open(path, "rb") as f:
        return pickle.load(f)
    
# ============================================================================ #

def scaleX(X, scaler=None, cols=('Time', 'Amount')):
    scaler = scaler or loadScaler()
    X = X.copy()
    X[list(cols)] = scaler.transform(X[list(cols)])
    return X

# ============================================================================ #

def loadModel(name, path=None):
    path = path or (MODELS_DIR / f"{name}.pkl")
    if not path.exists():
        raise FileNotFoundError(
            f"Model '{name}' not found at {path}."
        )
    print(f"Loading model from {path}")
    with open(path, "rb") as f:
        return pickle.load(f)

# ---------------------------------- CV step --------------------------------- #

def fitScaleX(X_fit, X_apply, cols=('Time', 'Amount')):
    # exclusive for cross validation
    scaler = StandardScaler().fit(X_fit[list(cols)])
    X_fit = X_fit.copy()
    X_apply = X_apply.copy()
    X_fit[list(cols)] = scaler.transform(X_fit[list(cols)])
    X_apply[list(cols)] = scaler.transform(X_apply[list(cols)])
    return scaler, X_fit, X_apply

# ============================================================================ #

def modelName(model):
    name = type(model).__name__
    params = model.get_params()
    if name == "KNeighborsClassifier":
        return f"{name}_k{params['n_neighbors']}"
    if name == "DecisionTreeClassifier":
        return f"{name}_d{params['max_depth']}"
    if name == "LogisticRegression":
        return f"{name}_C{params['C']}"
    return name

# ----------------------------------- main ----------------------------------- #

df = loader()

X = df.drop(columns='Class')
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y, 
    test_size=0.2,
    stratify=y,
    shuffle=True,
    random_state=42,
)