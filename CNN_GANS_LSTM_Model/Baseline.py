import os
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

data_dir = r"C:\..PhD Thesis\DataSet\GripForce_Regression"

y_train = np.load(os.path.join(data_dir, "y_train.npy")).reshape(-1)
y_test  = np.load(os.path.join(data_dir, "y_test.npy")).reshape(-1)

y_pred = np.full_like(y_test, y_train.mean())

print("Constant baseline (predict train mean)")
print("mean(y_train) =", float(y_train.mean()))
print("R2  =", float(r2_score(y_test, y_pred)))
print("MAE =", float(mean_absolute_error(y_test, y_pred)))
print("MSE =", float(mean_squared_error(y_test, y_pred)))