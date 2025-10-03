from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix

def soft_voting(logistic_model, xgb_model, mlp_model, X_test, y_test):
    
    log_proba = logistic_model.predict_proba(X_test)[:, 1]
    xgb_proba = xgb_model.predict_proba(X_test)[:, 1]
    mlp_proba = mlp_model.predict(X_test).ravel()  # خروجی sigmoid Keras

    
    weights = [1, 3, 2]  # وزن‌ها را می‌توان تغییر داد
    final_proba = (weights[0]*log_proba + weights[1]*xgb_proba + weights[2]*mlp_proba) / sum(weights)

    
    final_preds = (final_proba >= 0.5).astype(int)

    
    print("Accuracy:", accuracy_score(y_test, final_preds))
    print("Precision:", precision_score(y_test, final_preds))
    print("Recall:", recall_score(y_test, final_preds))
    print("F1-score:", f1_score(y_test, final_preds))
    print("ROC-AUC:", roc_auc_score(y_test, final_proba))
    print("Classification Report:\n", classification_report(y_test, final_preds))
    print("Confusion Matrix:\n", confusion_matrix(y_test, final_preds))