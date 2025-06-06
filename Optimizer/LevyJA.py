import pandas as pd
import numpy as npa
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, f1_score
from xgboost import XGBClassifier
import random
import scipy.stats as stats

train_data = pd.read_csv("Training_PCA.csv")
test_data = pd.read_csv("Testing_PCA.csv")

X_train = train_data.iloc[:, :-1]
y_train = train_data.iloc[:, -1]
X_test = test_data.iloc[:, :-1]
y_test = test_data.iloc[:, -1]

# Compute sample weights
classes = np.unique(y_train)
weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
class_weights = dict(zip(classes, weights))
sample_weights = np.array([class_weights[label] for label in y_train])

# ----------------------- Optuna Best Params (Initial Point) -----------------------
best_params = {
    'n_estimators': 300,
    'max_depth': 6,
    'learning_rate': 0.2567451389613149,
    'subsample': 0.8400092987532728,
    'colsample_bytree': 0.8464805154502861,
    'gamma': 0.046440916333720494,
    'reg_lambda': 1.6414868645406089,
    'reg_alpha': 0.49211449804115437
}

# ----------------------- Define Bounds Based on Best Params -----------------------
param_bounds = {
    'learning_rate': (best_params['learning_rate'] * 0.5, best_params['learning_rate'] * 1.5),
    'max_depth': (max(3, best_params['max_depth'] - 2), min(15, best_params['max_depth'] + 2)),
    'subsample': (best_params['subsample'] * 0.8, 1.0),
    'colsample_bytree': (best_params['colsample_bytree'] * 0.8, 1.0),
    'gamma': (0.0, best_params['gamma'] * 1.5),
    'reg_lambda': (max(1e-3, best_params['reg_lambda'] * 0.5), best_params['reg_lambda'] * 1.5),
    'reg_alpha': (max(1e-3, best_params['reg_alpha'] * 0.5), best_params['reg_alpha'] * 1.5),
    'n_estimators': (int(best_params['n_estimators'] * 0.8), best_params['n_estimators'])
}

# ----------------------- Levy Flight Generator -----------------------
def levy_flight(beta=1.5, size=1):
    sigma_u = (stats.gamma(1 + beta).pdf(1) * np.sin(np.pi * beta / 2) /
               (stats.gamma((1 + beta) / 2).pdf(1) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    u = np.random.normal(0, sigma_u, size=size)
    v = np.random.normal(0, 1, size=size)
    step = u / (np.abs(v) ** (1 / beta))
    return step

# ----------------------- Fitness Function (macro F1) -----------------------
def fitness_function(params):
    model = XGBClassifier(**params,
                          objective='multi:softmax',
                          num_class=len(np.unique(y_train)),
                          eval_metric='mlogloss',
                          use_label_encoder=False,
                          verbosity=0)
    model.fit(X_train, y_train, sample_weight=sample_weights)
    y_pred = model.predict(X_test)
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    return macro_f1

# ----------------------- Lévy JA Optimizer -----------------------
def levy_ja(fitness_func, param_bounds, num_agents=10, max_iter=20):
    agents = []
    for _ in range(num_agents):
        agent = {k: np.random.uniform(low, high) for k, (low, high) in param_bounds.items()}
        agent['max_depth'] = int(agent['max_depth'])
        agent['n_estimators'] = int(agent['n_estimators'])
        agents.append(agent)

    best_agent = max(agents, key=fitness_func)
    best_score = fitness_func(best_agent)

    for iteration in range(1, max_iter + 1):
        for i in range(num_agents):
            new_agent = {}
            for key in param_bounds.keys():
                step = levy_flight(beta=1.5, size=1)[0]
                val = agents[i][key] + step * (agents[i][key] - best_agent[key])
                low, high = param_bounds[key]
                if isinstance(low, int) or 'int' in key or key in ['max_depth', 'n_estimators']:
                    val = int(np.clip(val, low, high))
                else:
                    val = float(np.clip(val, low, high))
                new_agent[key] = val

            new_agent['max_depth'] = int(new_agent['max_depth'])
            new_agent['n_estimators'] = int(new_agent['n_estimators'])

            new_score = fitness_func(new_agent)
            if new_score > fitness_func(agents[i]):
                agents[i] = new_agent
                if new_score > best_score:
                    best_score = new_score
                    best_agent = new_agent

        print(f"Iteration {iteration} | Best Macro F1 Score: {best_score:.4f}")

    return best_agent, best_score

# ----------------------- Run Lévy JA Optimization -----------------------
print("Running Lévy JA optimization (objective: macro F1)...")
best_levy_params, best_macro_f1 = levy_ja(fitness_function, param_bounds, num_agents=10, max_iter=20)

# ----------------------- Train Final Model -----------------------
print("\nTraining final model with optimized parameters...")
model = XGBClassifier(**best_levy_params,
                      objective='multi:softmax',
                      num_class=len(np.unique(y_train)),
                      eval_metric='mlogloss',
                      use_label_encoder=False,
                      verbosity=0)
model.fit(X_train, y_train, sample_weight=sample_weights)

# ----------------------- Predictions -----------------------
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# ----------------------- Evaluation -----------------------
print("\n Classification Report - Train:")
print(classification_report(y_train, y_train_pred))

print("\nClassification Report - Test:")
print(classification_report(y_test, y_test_pred))

train_macro_f1 = f1_score(y_train, y_train_pred, average='macro')
test_macro_f1 = f1_score(y_test, y_test_pred, average='macro')

print(f"\n Train Macro F1 Score: {train_macro_f1:.4f}")
print(f" Test Macro F1 Score: {test_macro_f1:.4f}")

print("\n Best Parameters Found by Lévy JA:")
print(best_levy_params)
print(f"Best Macro F1 Score: {best_macro_f1:.4f}")
