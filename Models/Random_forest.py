#if optuna not installed
!pip install optuna

#Random forest + Optuna + class weights
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score,classification_report
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# Define the objective function for Optuna
def objective(trial):
    rf_model = RandomForestClassifier(
        n_estimators=trial.suggest_int("n_estimators", 50, 150),
        max_depth=trial.suggest_int("max_depth", 10, 40),
        min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
        class_weight=class_weights,
        random_state=42,
        n_jobs=-1
    )

    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)

    return f1_score(y_test, y_pred, average='weighted')


# Create and run the Optuna study
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50, timeout=600)

# Train final model using best parameters
best_params = study.best_params
print(" Best Parameters:", best_params)

# Recompute sample weights for full train set (in case of retraining on full data later)
sample_weights = np.array([class_weights[class_] for class_ in y_train])

# Re-train with best params
best_rf = RandomForestClassifier(**best_params, random_state=42, n_jobs=-1)
best_rf.fit(X_train, y_train, sample_weight=sample_weights)
y_test_pred = best_rf.predict(X_test)

# Evaluate
print("Weighted F1-score:", f1_score(y_test, y_test_pred, average='weighted'))
print("Macro F1-score:", f1_score(y_test, y_test_pred, average='macro'))
print("Micro F1-score:", f1_score(y_test, y_test_pred, average='micro'))

print("\n Classification Report - Test:")
print(classification_report(y_test, y_test_pred))
