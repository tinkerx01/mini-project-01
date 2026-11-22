from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import data_prep as dp

model_LR = LogisticRegression()
model_knn = KNeighborsClassifier(4, n_jobs=1)
model_dt = DecisionTreeClassifier(max_depth=4, max_features=10)

model_LR.fit(dp.X_train, dp.y_train)
model_knn.fit(dp.X_train, dp.y_train)
model_dt.fit(dp.X_train, dp.y_train)