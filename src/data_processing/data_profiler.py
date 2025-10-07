from __future__ import annotations
import pandas as pd
from ydata_profiling import ProfileReport

def make_profile_html(
    df: pd.DataFrame,
    title: str = "Data Profile",
    mode: str = "minimal",
    sample_max: int = 5000,
    correlations: bool = False,
) -> str:
    """
    Tworzy bezpieczny i szybki raport profilujący (HTML) dla danych.
    
    Parametry:
    - df: DataFrame do analizy
    - title: tytuł raportu
    - mode: 'minimal' | 'explorative' | 'full'
    - sample_max: maks. liczba wierszy (dla dużych danych)
    - correlations: czy liczyć korelacje (może być wolne)
    
    Zwraca: HTML (str)
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return "<p><strong>Brak danych do profilowania.</strong></p>"

    # Sampling (dla wydajności)
    if len(df) > sample_max:
        df = df.sample(sample_max, random_state=42)
        note = f"Użyto próbki {sample_max:,} wierszy z {len(df):,}."
    else:
        note = f"Profil pełnych danych ({len(df):,} wierszy)."

    # Tryb raportu
    minimal = mode == "minimal"
    explorative = mode == "explorative"

    # Opcje profilu
    try:
        profile = ProfileReport(
            df,
            title=title,
            minimal=minimal,
            explorative=explorative,
            correlations={
                "pearson": correlations,
                "spearman": correlations,
                "kendall": False,
                "phi_k": False,
                "cramers": False,
            } if correlations else None,
            progress_bar=False,
            pool_size=0,
            infer_dtypes=True,
            html={"style": {"full_width": True, "theme": "flatly"}},
        )

        html = profile.to_html()
        # Dodaj notkę o próbkowaniu na początku
        header = f"<p style='color:#888;font-size:0.9em'><em>{note}</em></p>"
        return header + html
    except Exception as e:
        return f"<p><strong>Błąd podczas generowania profilu:</strong> {e}</p>"
