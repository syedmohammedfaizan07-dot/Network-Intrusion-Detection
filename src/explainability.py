import shap
import pandas as pd
import numpy as np


def run_shap_explainability(model, X_train, X_test, feature_names):
    print("\n--- Computing XAI-SHAP Feature Importance ---")

    # Background summary sample for kernel explainability
    background = X_train[np.random.choice(X_train.shape[0], 50, replace=False)]
    explainer = shap.KernelExplainer(model.predict_proba, background)

    # Explain a subset of test instances
    shap_values = explainer.shap_values(X_test[:20])

    # Convert multi-class/3D array shapes into a clean 1D array per feature
    if isinstance(shap_values, list):
        importances = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
    else:
        importances = np.abs(shap_values).mean(axis=(0, -1)) if shap_values.ndim == 3 else np.abs(shap_values).mean(
            axis=0)

    # Flatten to guarantee 1-dimensional array shape
    importances = np.ravel(importances)

    importance_df = pd.DataFrame({
        'Feature': list(feature_names),
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    print(importance_df.head(5).to_string(index=False))
    return importance_df