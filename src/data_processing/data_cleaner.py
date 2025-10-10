import pandas as pd, numpy as np
def clean_dataframe(df: pd.DataFrame)->pd.DataFrame:
    df=df.copy()
    for c in df.select_dtypes(include=["object","string"]).columns: df[c]=df[c].astype("string").str.strip()
    num=df.select_dtypes(include=[np.number]).columns
    df[num]=df[num].replace([np.inf,-np.inf], np.nan)
    for c in num:
        if df[c].isna().any(): df[c]=df[c].fillna(df[c].median())
    for c in df.select_dtypes(include=["object","string"]).columns:
        if df[c].isna().any(): df[c]=df[c].fillna("missing")
    return df.drop_duplicates()
