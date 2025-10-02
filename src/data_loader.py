import os
import numpy as np
import pandas as pd

def load_and_preprocess():

    BASE_DIR = os.path.dirname(os.path.dirname(__file__))
    data_path = os.path.join(BASE_DIR, "adult", "adult.data")
    test_path = os.path.join(BASE_DIR, "adult", "adult.test")

    column_names = [
        "age", "workclass", "fnlwgt", "education", "education_num",
        "marital_status", "occupation", "relationship", "race", "sex",
        "capital_gain", "capital_loss", "hours_per_week", "native_country", "income"
    ]

    df_train = pd.read_csv(data_path, header=None, names=column_names)
    df_test = pd.read_csv(test_path, header=None, names=column_names)

    df_train = df_train.apply(lambda x: x.str.strip() if x.dtype == "object" else x)   ### for detect " ?" and "? " and "?"
    df_test  = df_test.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

    df_train.replace('?', np.nan, inplace=True) 
    df_test.replace('?', np.nan, inplace=True)

    # ### Dropping NaN values due to the small number of rows containing NaN compared to the total dataset.

    df_train.dropna(inplace=True)
    df_test.dropna(inplace=True)

    df_train['income'] = df_train['income'].str.strip().str.replace('.', '', regex=False)   ### Remove trailing '.' in 'income' values to unify categories before OHE
    df_test['income'] = df_test['income'].str.strip().str.replace('.', '', regex=False)

    categorical_cols = [
        "workclass", "education", "marital_status", "occupation",
        "relationship", "race", "sex", "native_country", "income"
    ]

    combined = pd.concat([df_train, df_test], axis=0)
    combined_onehot = pd.get_dummies(combined, columns=categorical_cols)   #, drop_first=True)   for linear models

    df_train_onehot = combined_onehot.iloc[:len(df_train), :].copy()
    df_test_onehot = combined_onehot.iloc[len(df_train):, :].copy()

    bool_cols_train = df_train_onehot.select_dtypes('bool').columns
    df_train_onehot[bool_cols_train] = df_train_onehot[bool_cols_train].astype('int64')

    bool_cols_test = df_test_onehot.select_dtypes('bool').columns
    df_test_onehot[bool_cols_test] = df_test_onehot[bool_cols_test].astype('int64')
    
    return df_train, df_test, df_train_onehot, df_test_onehot




