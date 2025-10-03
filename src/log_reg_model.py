from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

def log_reg_model(X_train, Y_train, X_val, Y_val, class_weight=None):
    log_reg = LogisticRegression(
        class_weight=class_weight,
        max_iter=500,
        random_state=42
    )
    log_reg.fit(X_train, Y_train)

    y_pred = log_reg.predict(X_val)

    print(classification_report(Y_val, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(Y_val, y_pred))