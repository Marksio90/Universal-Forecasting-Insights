import pandas as pd
import plotly.express as px

def histogram(df: pd.DataFrame, col: str):
    return px.histogram(df, x=col)

def scatter(df: pd.DataFrame, x: str, y: str):
    return px.scatter(df, x=x, y=y, trendline="ols")

def line(df: pd.DataFrame, x: str, y: str):
    return px.line(df, x=x, y=y)
