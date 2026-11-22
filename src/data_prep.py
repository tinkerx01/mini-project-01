import os
import shutil
import pandas as pd
import kagglehub
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

DATA_DIR = "data"
CSV_PATH = os.path.join(DATA_DIR, "creditcard.csv")

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


print(f"Number of Samples: {len(df)}")
print(f"Number of features: {df.shape[1]-1}") #'-1' is the Class column

labels = {0: "Normal", 1: "Fraud"}
count = df["Class"].value_counts()
for cls, n in count.items():
    print(f"{labels[cls]} (Class {cls}): {n}  ({n/len(df)*100:.3f}%)")

#sanity checks
print(f"inf Values: {np.isinf(df.select_dtypes(include=np.number)).sum().sum()}") #infs
print(f"null Values: {df.isnull().sum().sum()}") #nulls
print(f"Min Time: {df['Time'].min()}") #negative elapsed time
print(f"Min Amount: {df['Amount'].min()}") #negative Amount 
print(f"Min Class: {df['Class'].min()}") #negative class 
print(f"Is time column acsending: {df['Time'].is_monotonic_increasing}") #time column is ascending? (could've read the decription but just incase)
print(df.drop(columns=['Time', 'Amount', 'Class']).mean().describe()) #col vlues distribution (again could've read the docs)

X = df.drop(columns='Class')
y = df['Class']


X_test, X_train, y_train, y_test = train_test_split(
    X,
    y,
    random_state=42,
    stratify=y,
    shuffle=True,
    test_size=0.2
)
scaler = StandardScaler()
X_train[['Time', 'Amount']] = scaler.fit_transform(X_train[['Time', 'Amount']])
X_test[['Time', 'Amount']] = scaler.transform(X_test[['Time', 'Amount']])