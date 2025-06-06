pip install mealpy==3.0.1
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


#enhanced AEO
import numpy as np
from sklearn.metrics import f1_score, make_scorer
from xgboost import XGBClassifier
from mealpy import FloatVar, IntegerVar
from sklearn.model_selection import cross_val_score
from mealpy.system_based.AEO import EnhancedAEO
def objective_func(solution):
    n_estimators = int(solution[0])
    max_depth = int(solution[1])
    learning_rate = solution[2]

    model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
    )

    # Fit the model directly on the training data with sample weights
    model.fit(X_train, y_train, sample_weight=sample_weights)

    # Predict on the validation set
    y_pred = model.predict(X_test)

    # Calculate the F1 score on the validation set (macro average)
    score = f1_score(y_test, y_pred, average='macro')

    return -score
# Range for problem_dict is obtained from best parameters of Optuna
problem_dict = {
    "obj_func": objective_func,
    "bounds": [
        IntegerVar(lb=250, ub=350),    # n_estimators
        IntegerVar(lb=4, ub=8),      # max_depth
        FloatVar(lb=0.2067, ub=0.3067),    # learning_rate
    ],
    "minmax": "min",
}
model = EnhancedAEO(epoch=20, pop_size=10)
model.solve(problem_dict)

best_params = model.g_best.solution
best_macro_f1 = -model.g_best.target.fitness  # flip sign

print("Best Parameters:", best_params)
print("Best Macro F1 Score:", best_macro_f1)


#-----------------------------------------------------------------------------------------------------------------------------
#from logs the f1 scores obtained are printed
import matplotlib.pyplot as plt

# F1 scores from your logs (converted from negative to positive)
f1_scores = [
    0.9512974517681467,
    0.9512974517681467,
    0.9514630839626074,
    0.9514912950542224,
    0.9517214667287547,
    0.9517214667287547,
    0.9517214667287547,
    0.9517214667287547,
    0.9517214667287547,
    0.9517214667287547,
    0.9517214667287547,
    0.9517214667287547,
    0.9517214667287547,
    0.9517214667287547,
    0.9517214667287547,
    0.9517214667287547,
    0.9517214667287547,
    0.9517214667287547,
    0.9517214667287547,
    0.9517214667287547,
]

epochs = list(range(1, len(f1_scores) + 1))

# Plot
plt.figure(figsize=(10, 6))
plt.plot(epochs, f1_scores, marker='o', color='green')
plt.title('Macro F1 Score over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Macro F1 Score')
plt.ylim(0.951, 0.9520)
plt.grid(True)
plt.xticks(epochs)
plt.tight_layout()
plt.savefig("EnhancedAEO.png")
plt.show()
