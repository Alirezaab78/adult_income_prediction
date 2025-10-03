from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt

def xgboost_model(X_train, Y_train, X_val, Y_val):

    # محاسبه scale_pos_weight بر اساس کلاس‌های نامتوازن
    scale_pos_weight_value = (Y_train.shape[0] - sum(Y_train)) / sum(Y_train)

    xgb_base = XGBClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight_value,
        use_label_encoder=False,
        eval_metric='logloss'  # برای جلوگیری از warning
    )


    param_grid = {
        'max_depth': [4, 5, 6],
        'learning_rate': [0.05, 0.1],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
        'scale_pos_weight': [None, scale_pos_weight_value]
    }


    grid_search = GridSearchCV(
        estimator=xgb_base,
        param_grid=param_grid,
        scoring='f1',       # می‌تونی 'recall' یا 'precision' هم بذاری
        cv=3,               # 3-fold cross validation
        verbose=2
    )

    grid_search.fit(X_train, Y_train)


    print("Best Parameters:", grid_search.best_params_)
    print("Best F1 Score (CV):", grid_search.best_score_)


    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_val)


    print("\n=== Classification Report (Validation) ===")
    print(classification_report(Y_val, y_pred))


    cm = confusion_matrix(Y_val, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues")
    plt.title("Confusion Matrix (Best XGBoost)")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()

    print("Classification Report:\n", classification_report(Y_val, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(Y_val, y_pred))
    print("ROC-AUC Score:", roc_auc_score(Y_val, y_pred))
