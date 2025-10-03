def scale_test(df_test_onehot, scaler):
    numeric_cols = ['age', 'fnlwgt', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week']
    
    df_test_scaled = df_test_onehot.copy()
    
    df_test_scaled[numeric_cols] = scaler.transform(df_test_onehot[numeric_cols])
    
    X_test = df_test_scaled.drop(columns=['income_<=50K', 'income_>50K'])
    y_test = df_test_scaled['income_>50K']
    
    return X_test, y_test