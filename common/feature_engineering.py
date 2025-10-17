import pandas as pd, numpy as np
from sklearn.feature_selection import mutual_info_classif

def add_interaction_features(df: pd.DataFrame):
    X = df.copy()
    if {'credit_utilization','balance_to_income'}.issubset(X.columns):
        X['credit_x_balance'] = X['credit_utilization'] * X['balance_to_income']
    return X

def add_decay_features(df: pd.DataFrame):
    X = df.copy()
    if {'recent_loan_number','recent_account_months'}.issubset(X.columns):
        X['loan_decay'] = np.exp(-X['recent_loan_number'] / (X['recent_account_months'] + 1))
    return X

def feature_selection(X: pd.DataFrame, y: pd.Series, top_k=50):
    mi = mutual_info_classif(X.fillna(0), y)
    mi_df = pd.DataFrame({'feature': X.columns, 'mi': mi})
    selected = mi_df.sort_values('mi', ascending=False).head(top_k)['feature']
    return X[selected]
