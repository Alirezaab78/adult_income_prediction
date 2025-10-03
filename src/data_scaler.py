import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from data_loader import load_and_preprocess

def standard_scaler_and_validation():

    df_train, df_test, df_train_onehot, df_test_onehot = load_and_preprocess()

    df_train_onehot_split, df_val_onehot_split = train_test_split(
        df_train_onehot, test_size=0.2, stratify=df_train_onehot['income_>50K'], random_state=42
    )


    numeric_cols = ['age','fnlwgt','education_num','capital_gain','capital_loss','hours_per_week']

    scaler = StandardScaler()

    df_train_onehot_split_scaled = df_train_onehot_split.copy()
    df_val_onehot_split_scaled = df_val_onehot_split.copy()

    df_train_onehot_split_scaled[numeric_cols] = scaler.fit_transform(df_train_onehot_split[numeric_cols])
    df_val_onehot_split_scaled[numeric_cols] = scaler.transform(df_val_onehot_split[numeric_cols])


    X_train = df_train_onehot_split_scaled.drop(columns=['income_<=50K', 'income_>50K'])
    Y_train = df_train_onehot_split_scaled['income_>50K']

    X_val = df_val_onehot_split_scaled.drop(columns=['income_<=50K', 'income_>50K'])
    Y_val = df_val_onehot_split_scaled['income_>50K']

    return X_train, Y_train, X_val, Y_val, scaler

