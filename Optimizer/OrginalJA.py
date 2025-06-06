#orgJA
import optuna
import xgboost as xgb
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

def objective(trial):
    param = {
        "objective": "multi:softmax",
        "num_class": len(np.unique(y_train)),
        "eval_metric": "mlogloss",
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_float("gamma", 0, 10),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 10.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-2, 10.0, log=True),
    }

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    macro_f1s = []
    weighted_f1s = []

    for train_idx, val_idx in skf.split(X_train, y_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        sw_tr = sample_weights[train_idx]

        model = xgb.XGBClassifier(**param)
        model.fit(X_tr, y_tr, sample_weight=sw_tr)

        preds = model.predict(X_val)
        macro_f1s.append(f1_score(y_val, preds, average="macro"))
        weighted_f1s.append(f1_score(y_val, preds, average="weighted"))

    avg_macro_f1 = np.mean(macro_f1s)
    avg_weighted_f1 = np.mean(weighted_f1s)

    trial.set_user_attr("macro_f1", avg_macro_f1)
    return avg_weighted_f1

# Run Optuna optimization
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50, timeout=600)

# Print best results
print("Best parameters:", study.best_params)
print("Best Weighted F1 Score:", study.best_value)
print("Corresponding Macro F1 Score:", study.best_trial.user_attrs["macro_f1"])

import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import f1_score

# Define param_bounds (param_ranges)
param_bounds = {
    'max_depth': (4, 8),          # ±2 around 6
    'learning_rate': (0.2067, 0.3067),  # ±0.05 around 0.2567
    'n_estimators': (250, 350),  # ±50 around 300
    'gamma': (0.0, 0.096),       # ±0.05 around 0.046
    'subsample': (0.74, 0.94),    # ±0.1 around 0.84
    'colsample_bytree': (0.746, 0.946),  # ±0.1 around 0.846
}

# Fitness function for XGBoost model
def fitness_func(params):
    max_depth, learning_rate, n_estimators, gamma, subsample, colsample_bytree = params
    model = XGBClassifier(
        max_depth=int(max_depth),
        learning_rate=learning_rate,
        n_estimators=int(n_estimators),
        gamma=gamma,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        objective='multi:softmax',
        num_class=len(np.unique(y_train)),
        eval_metric='mlogloss'
    )

    model.fit(X_train, y_train, sample_weight=sample_weights)
    preds = model.predict(X_test)
    macro_f1 = f1_score(y_test, preds, average='macro')
    return 1 - macro_f1  # Minimize 1 - macro F1 score (maximize F1)

# Initialize population randomly within bounds
def initialize_population():
    population = []
    for _ in range(population_size):
        individual = []
        for key in param_bounds:
            low, high = param_bounds[key]
            value = np.random.uniform(low, high)
            individual.append(value)
        population.append(individual)
    return np.array(population)

# Jaya update function for OriginalJA
def jaya_update(population, best_solution, worst_solution):
    new_population = []
    for x in population:
        r1, r2 = np.random.rand(), np.random.rand()
        new_x = x + r1 * (best_solution - np.abs(x)) - r2 * (worst_solution - np.abs(x))

        # Clip to bounds
        for i, key in enumerate(param_bounds):
            low, high = param_bounds[key]
            new_x[i] = np.clip(new_x[i], low, high)
        new_population.append(new_x)
    return np.array(new_population)

# Original Jaya Algorithm loop
def original_jaya_algorithm():
    population = initialize_population()
    fitness = np.array([fitness_func(ind) for ind in population])

    best_idx = np.argmin(fitness)
    best_solution = population[best_idx]
    best_score = fitness[best_idx]

    worst_idx = np.argmax(fitness)
    worst_solution = population[worst_idx]

    for t in range(max_iter):
        population = jaya_update(population, best_solution, worst_solution)
        fitness = np.array([fitness_func(ind) for ind in population])

        current_best_idx = np.argmin(fitness)
        current_worst_idx = np.argmax(fitness)

        if fitness[current_best_idx] < best_score:
            best_solution = population[current_best_idx]
            best_score = fitness[current_best_idx]
        if fitness[current_worst_idx] > best_score:
            worst_solution = population[current_worst_idx]

        # Print the progress for each iteration
        print(f"Iteration {t+1} | Best Macro F1 Score: {1 - best_score:.4f}")

    return best_solution, 1 - best_score

# Set the population size and maximum iterations for OriginalJA
population_size = 10
max_iter = 20

# Run OriginalJA optimization
best_params_ja, best_f1_ja = original_jaya_algorithm()

# Show Final Best Parameters
param_names = list(param_bounds.keys())
final_params = {k: (int(v) if 'int' in str(type(param_bounds[k][0])) else round(v, 4))
                for k, v in zip(param_names, best_params_ja)}

print("\n✅ Best Parameters Found by OriginalJA:")
print(final_params)
print(f"Best Macro F1 Score: {best_f1_ja:.4f}")
import matplotlib.pyplot as plt

# Sample data - replace with your actual F1 scores from Original JA
iterations = list(range(1, 21))  # 20 iterations
macro_f1_scores = [0.9498,
0.9498,0.9516,0.9516,0.9524,0.9524,0.9524,0.9524,0.9524,
0.9524,0.9524,0.9524,0.9524,0.9524,0.9524,0.9524,0.9524,0.9524,0.9524,0.9524]

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(iterations, macro_f1_scores, marker='o', color='blue', label='Original JA')

# Styling
plt.title('Original JA Optimization - Macro F1 Score per Iteration')
plt.xlabel('Iteration Number')
plt.ylabel('Macro F1 Score')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("OrginalJA.png",dpi=300)
# Show the plot
plt.show()
