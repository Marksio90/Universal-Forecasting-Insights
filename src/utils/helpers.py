import pandas as pd
def smart_read(path_or_buffer, **kwargs) -> pd.DataFrame:
    name = getattr(path_or_buffer, "name", str(path_or_buffer)).lower()
    if name.endswith(".csv"): return pd.read_csv(path_or_buffer, **kwargs)
    if name.endswith(".parquet"): return pd.read_parquet(path_or_buffer, **kwargs)
    if name.endswith(".xlsx") or name.endswith(".xls"): return pd.read_excel(path_or_buffer, **kwargs)
    return pd.read_csv(path_or_buffer, **kwargs)
