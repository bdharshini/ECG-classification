import numpy as np
import sklearn
import scipy
import pandas as pd
import pandas as pd
train_data = pd.read_csv("Training_PCA.csv")
test_data = pd.read_csv("Testing_PCA.csv")

# Split features and labels
X_train = train_data.iloc[:, :-1]  # All columns except the last one
y_train = train_data.iloc[:, -1]   # Last column as target
X_test = test_data.iloc[:, :-1]
y_test = test_data.iloc[:, -1]
print(f"Train:{X_train.shape}")
print(f"Train:{X_test.shape}")
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# Compute class weights
classes = np.unique(y_train)
weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
class_weights = dict(zip(classes, weights))

# Compute sample weights for each class
sample_weights = np.array([class_weights[class_] for class_ in y_train])
print(sample_weights)
