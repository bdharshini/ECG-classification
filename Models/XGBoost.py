#XGBoost + Optuna + class weights
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
from sklearn.metrics import classification_report, f1_score
from xgboost import XGBClassifier
best_params = {'n_estimators': 300,
               'max_depth': 6,
               'learning_rate': 0.2567451389613149,
              'subsample': 0.8400092987532728,
              'colsample_bytree': 0.8464805154502861,
              'gamma': 0.046440916333720494,
              'reg_lambda': 1.6414868645406089,
              'reg_alpha': 0.49211449804115437}
model = XGBClassifier(**best_params)
model.fit(X_train, y_train, sample_weight=sample_weights)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

print("📊 Classification Report - Train (SMOTE applied):")
print(classification_report(y_train, y_train_pred))
print("\n📊 Classification Report - Test:")
print(classification_report(y_test, y_test_pred))

train_macro_f1 = f1_score(y_train, y_train_pred, average='macro')
test_macro_f1 = f1_score(y_test, y_test_pred, average='macro')
print(f"\n Train Macro F1 Score: {train_macro_f1:.4f}")
print(f" Test Macro F1 Score: {test_macro_f1:.4f}")
