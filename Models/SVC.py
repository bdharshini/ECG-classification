import optuna
from sklearn.svm import SVC
from sklearn.metrics import f1_score, classification_report

# Set Optuna logging level to DEBUG
optuna.logging.set_verbosity(optuna.logging.DEBUG)

# Define the objective function for Optuna
def objective(trial):
    print(f"\nStarting Trial {trial.number}")

    # Suggest hyperparameters
    C = trial.suggest_float("C", 0.1, 10.0, log=True)
    kernel = trial.suggest_categorical("kernel", ["rbf", "sigmoid", "poly"])
    gamma = trial.suggest_categorical("gamma", ["scale", "auto"])

    print(f"Suggested Parameters -> C: {C:.4f}, kernel: {kernel}, gamma: {gamma}")

    try:
        # Initialize and fit the model
        model = SVC(C=C, kernel=kernel, gamma=gamma, random_state=42)
        model.fit(X_train, y_train, sample_weight=sample_weights)

        # Predictions and scoring
        preds = model.predict(X_test)
        score = f1_score(y_test, preds, average='macro')
        print(f"Trial {trial.number} F1 Score: {score:.4f}")

        return score

    except Exception as e:
        print(f"Trial {trial.number} failed with error: {e}")
        raise

# Create and run the Optuna study
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30, timeout=600)

# Best model
print("\nBest Params for SVC:", study.best_params)

# Train best model with optimized parameters and sample weights
best_svc = SVC(**study.best_params, random_state=42)
best_svc.fit(X_train, y_train, sample_weight=sample_weights)
svc_preds = best_svc.predict(X_test)
svc_preds_train = best_svc.predict(X_train)

# Final Evaluation
svc_f1_test = f1_score(y_test, svc_preds, average='macro')
print(f"SVC Macro Test F1 Score: {svc_f1_test:.4f}")
svc_f1_train = f1_score(y_train, svc_preds_train, average='macro')
print(f"SVC Macro Train F1 Score: {svc_f1_train:.4f}")

# Print Classification Report
print("Classification Report:\n", classification_report(y_test, svc_preds))
