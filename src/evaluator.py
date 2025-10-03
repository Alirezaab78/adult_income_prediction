from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import numpy as np

def evaluate_classification(model, X_val, y_val, threshold=0.5, is_logits=False):
    """
    Evaluate a classification model using Classification Report, Confusion Matrix, and ROC-AUC Score.

    Parameters:
    - model: trained model (must have .predict or .predict_proba)
    - X_val: validation features
    - y_val: validation target
    - threshold: decision threshold for binary classification
    - is_logits: if True, assumes model outputs logits and applies sigmoid

    Returns:
    None (prints results)
    """
    # پیش‌بینی احتمال
    if hasattr(model, "predict_proba"):
        y_pred_prob = model.predict_proba(X_val)[:, 1]
    else:
        y_pred_prob = model.predict(X_val).ravel()

    # اگر logits بودند، سیگموید بزن
    if is_logits:
        y_pred_prob = 1 / (1 + np.exp(-y_pred_prob))

    # بر اساس threshold کلاس بده
    y_pred_class = (y_pred_prob >= threshold).astype(int)

    print("Classification Report:\n", classification_report(y_val, y_pred_class))
    print("Confusion Matrix:\n", confusion_matrix(y_val, y_pred_class))
    print("ROC-AUC Score:", roc_auc_score(y_val, y_pred_prob))
