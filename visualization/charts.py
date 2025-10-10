import plotly.express as px, pandas as pd
def corr_heatmap(df: pd.DataFrame):
    num=df.select_dtypes(include="number")
    if num.shape[1]<2: return None
    corr=num.corr(numeric_only=True)
    fig=px.imshow(corr, aspect="auto"); fig.update_layout(margin=dict(l=10,r=10,b=10,t=28))
    return fig
