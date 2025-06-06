import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, make_scorer
from xgboost import XGBClassifier
from mealpy import FloatVar, IntegerVar
from sklearn.model_selection import cross_val_score
from mealpy.evolutionary_based.DE import JADE


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

    return score

problem_dict = {
    "obj_func": objective_func,
    "bounds": [
        IntegerVar(lb=250, ub=350),    # n_estimators
        IntegerVar(lb=4, ub=8),      # max_depth
        FloatVar(lb=0.2067, ub=0.3067),    # learning_rate
    ],
    "minmax": "max",
}
model = JADE(epoch=20, pop_size=10)
model.solve(problem_dict)

best_params = model.g_best.solution
best_macro_f1 = model.g_best.target.fitness  # flip sign

print("Best Parameters:", best_params)
print("Best Macro F1 Score:", best_macro_f1)
import matplotlib.pyplot as plt

# Extracted F1 scores from the JADE log
f1_scores = [
    0.9493475848498999,
    0.9493475848498999,
    0.9516667449690784,
    0.9516667449690784,
    0.9516667449690784,
    0.9516667449690784,
    0.9516811576171575,
    0.9516811576171575,
    0.9516811576171575,
    0.9516811576171575,
    0.9516811576171575,
    0.9516811576171575,
    0.9516811576171575,
    0.9516811576171575,
    0.9516811576171575,
    0.9516811576171575,
    0.9516811576171575,
    0.9516811576171575,
    0.9516811576171575,
    0.952600968671377,
]

epochs = list(range(1, len(f1_scores) + 1))

# Plot
plt.figure(figsize=(10, 6))
plt.plot(epochs, f1_scores, marker='o', color='blue')
plt.title('JADE: Macro F1 Score over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Macro F1 Score')
plt.ylim(min(f1_scores) - 0.0005, max(f1_scores) + 0.0005)
plt.grid(True)
plt.xticks(epochs)
plt.tight_layout()
plt.savefig("JADE.png")
plt.show()
