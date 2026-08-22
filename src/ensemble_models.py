from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

def train_ensembles(X_train, X_test, y_train, y_test, class_names):
    print("\n--- Training Random Forest Classifier ---")
    rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    print(f"Random Forest Accuracy: {accuracy_score(y_test, rf_preds):.4f}")
    print(classification_report(y_test, rf_preds, target_names=class_names, zero_division=0))

    print("\n--- Training Gradient Boosting Classifier ---")
    gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
    gb.fit(X_train, y_train)
    gb_preds = gb.predict(X_test)
    print(f"Gradient Boosting Accuracy: {accuracy_score(y_test, gb_preds):.4f}")
    print(classification_report(y_test, gb_preds, target_names=class_names, zero_division=0))

    joblib.dump(rf, 'models/random_forest.joblib')
    joblib.dump(gb, 'models/gradient_boosting.joblib')

    return rf, gb