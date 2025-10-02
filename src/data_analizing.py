import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from data_loader import load_and_preprocess

def analize_data():

    df_train, df_test, df_train_onehot, df_test_onehot = load_and_preprocess()

    income_counts = df_train['income'].value_counts()
    income_counts.plot(kind='bar', color=['#4CAF50', '#FF5722'])
    plt.title('Income Class Distribution')
    plt.xlabel('Income')
    plt.ylabel('Count')
    plt.show()

    print(income_counts / len(df_train) * 100)




    numeric_cols = ['age','fnlwgt','education_num','capital_gain','capital_loss','hours_per_week']

    target_binary = df_train_onehot['income_>50K']

    correlations = df_train_onehot[numeric_cols + ['income_>50K']].corr()

    plt.figure(figsize=(8,6))
    sns.heatmap(correlations, annot=True, cmap='coolwarm')
    plt.title('Pearson Correlation with Target')
    plt.show()



    categorical_cols = ['workclass', 'education', 'marital_status', 'occupation', 
                        'relationship', 'race', 'sex', 'native_country']

    for col in categorical_cols:
        ct = pd.crosstab(df_train[col], df_train['income'], normalize='index')
        ct = ct.reset_index()  # برای استفاده در seaborn باید دیتافریم معمولی باشه
        ct_melted = ct.melt(id_vars=col, value_vars=ct.columns[1:], 
                            var_name='Income', value_name='Proportion')

        plt.figure(figsize=(8,4))
        sns.barplot(data=ct_melted, x=col, y='Proportion', hue='Income')
        plt.title(f'Income distribution by {col}')
        plt.xticks(rotation=45)
        plt.legend(title='Income')
        plt.tight_layout()
        plt.show()


