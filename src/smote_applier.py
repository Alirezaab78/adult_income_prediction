
from imblearn.over_sampling import SMOTE

def smote_apply(X_train, Y_train):
    smote = SMOTE(random_state=42)


    X_train_res, Y_train_res = smote.fit_resample(X_train, Y_train)

    print("قبل SMOTE:", X_train.shape, sum(Y_train==1), sum(Y_train==0))
    print("بعد SMOTE:", X_train_res.shape, sum(Y_train_res==1), sum(Y_train_res==0))
    return X_train_res, Y_train_res