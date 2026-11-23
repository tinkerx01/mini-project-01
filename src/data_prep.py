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
    print(f"{CSV_PATH} missing or corrupted, downloading...")
    os.makedirs(DATA_DIR, exist_ok=True)

    dataset_path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
    src = os.path.join(dataset_path, "creditcard.csv")

    shutil.copy(src, CSV_PATH)
    print(f"Saved to {CSV_PATH}")

#handling existing dataset so it takes less time.
if is_valid_csv(CSV_PATH):
    print(f"Found valid file at {CSV_PATH}, skipping download.")
else:
  loader()

df = pd.read_csv(CSV_PATH)

# df.info()
# df.head()
def statistics():
    print(f"Number of Samples: {len(df)}")
    print(f"Number of features: {df.shape[1]-1}") #'-1' is the Class column

#========================================================================

def distribution():
    labels = {0: "Normal", 1: "Fraud"}
    count = df["Class"].value_counts()
    for cls, n in count.items():
        print(f"{labels[cls]} (Class {cls}): {n}  ({n/len(df)*100:.3f}%)")

#========================================================================

#sanity checks
def MissingValues():
    print(f"inf Values: {np.isinf(df.select_dtypes(include=np.number)).sum().sum()}") #infs
    print(f"null Values: {df.isnull().sum().sum()}") #nulls
    print(f"Min Time: {df['Time'].min()}") #negative elapsed time
    print(f"Min Amount: {df['Amount'].min()}") #negative Amount 
    print(f"Min Class: {df['Class'].min()}") #negative class 
    print(f"Is time column acsending: {df['Time'].is_monotonic_increasing}") #time column is ascending? (could've read the decription but just incase)
    print(df.drop(columns=['Time', 'Amount', 'Class']).mean().describe()) #col vlues distribution (again could've read the docs)

#========================================================================

def duplicates():
    print(f"Number of duplicated samples: {df.duplicated().sum()}")
    dupes = df[df.duplicated(keep=False)].sort_values(list(df.columns))
    print(dupes['Class'].value_counts())

#========================================================================


X = df.drop(columns='Class')
y = df['Class']


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    random_state=42,
    stratify=y,
    shuffle=True,
    test_size=0.2
)

scaler = StandardScaler()
scaler.fit(X_train[['Time', 'Amount']])

def saveScaler(scaler):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / "scaler.pkl"
    with open(path, "wb") as f:
        pickle.dump(scaler, f)