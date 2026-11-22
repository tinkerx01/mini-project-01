from sklearn.metrics import accuracy_score
import train as t
import data_prep as dp


y_pred_LR = t.model_LR.predict(dp.X_test)
y_pred_knn = t.model_knn.predict(dp.X_test)
y_pred_dt = t.model_dt.predict(dp.X_test)

LR_acc = accuracy_score(y_pred_LR, dp.y_test)
knn_acc = accuracy_score(y_pred_knn, dp.y_test)
dt_acc = accuracy_score(y_pred_dt, dp.y_test)

print(f"Accuracy for LR: {LR_acc}")
print(f"Accuracy for knn: {knn_acc}")
print(f"Accuracy for decision tree: {dt_acc}")

