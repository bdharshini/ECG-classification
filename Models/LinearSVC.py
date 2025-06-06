#Linear SVC+ Optuna + class weights
import optuna
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

def objective(trial):
    # Suggest the regularization parameter C and class weights
    C = trial.suggest_float("linearsvc__C", 0.01, 10.0, log=True)

    # Build the pipeline with class weights
    pipeline = make_pipeline(
        StandardScaler(),
        LinearSVC(C=C, max_iter=10000, dual=False, class_weight=class_weights)
    )

    # Evaluate using cross-validation (macro F1)
    macro_f1 = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="f1_macro", n_jobs=-1).mean()
    return macro_f1

# Run the optimization
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30, timeout=300)

# Train with best parameters
best_C = study.best_params["linearsvc__C"]

best_pipeline = make_pipeline(
    StandardScaler(),
    LinearSVC(C=best_C, max_iter=10000, dual=False, class_weight=class_weights)
)

best_pipeline.fit(X_train, y_train)

# Predictions & Evaluation
linear_preds = best_pipeline.predict(X_test)

# Print classification report
print("Best Params for LinearSVC:", study.best_params)
print("Classification Report:\n", classification_report(y_test, linear_preds, target_names=np.unique(y_test).astype(str)))
