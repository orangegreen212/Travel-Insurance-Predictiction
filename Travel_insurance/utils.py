import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

# Настройки стиля (как в вашем проекте)
BLUE, ORANGE, GREEN = "#2563EB", "#F59E0B", "#10B981"

def clean_and_prepare(df):
    """Drops duplicates and fixes numeric formats in AnnualIncome."""
    df_clean = df.drop_duplicates().reset_index(drop=True)
    
    if 'AnnualIncome' in df_clean.columns:
        s = df_clean['AnnualIncome'].astype(str)
        s = s.str.replace(r'[\s\xa0]+', '', regex=True)
        s = s.str.replace(',', '.')
        s = s.str.replace(r'[^0-9.]', '', regex=True)
        df_clean['AnnualIncome'] = pd.to_numeric(s, errors='coerce')
        
    return df_clean

def encode_categorical(df, cat_cols):

    df_enc = df.copy()
    le = LabelEncoder()
    for col in cat_cols:
        df_enc[col] = le.fit_transform(df_enc[col])
    return df_enc

def plot_numerical_eda(df, target, num_cols):

    fig, axes = plt.subplots(1, len(num_cols), figsize=(15, 5))
    for ax, col in zip(axes, num_cols):
        sns.boxplot(data=df, x=target, y=col, ax=ax, palette=[BLUE, ORANGE])
        ax.set_title(col, fontweight="bold")
    plt.suptitle("Box-plots by Insurance Status", fontweight="bold")
    plt.tight_layout()
    plt.show()


def compare_models(models, X, y, cv):

    results = {}
    for name, model in models.items():
        auc = cross_val_score(model, X, y, cv=cv, scoring="roc_auc").mean()
        acc = cross_val_score(model, X, y, cv=cv, scoring="accuracy").mean()
        f1 = cross_val_score(model, X, y, cv=cv, scoring="f1").mean()
        results[name] = {"ROC-AUC": round(auc, 4), "Accuracy": round(acc, 4), "F1": round(f1, 4)}
    return pd.DataFrame(results).T