# === DATA_LOADER_PRO+++ ===
from __future__ import annotations
from typing import Optional, Tuple, Any, Literal, overload
import logging
import pandas as pd

# smart_read robi: autodetekcję formatu/enkodowania, normalizację NaN,
# downcast liczb, inteligentny parse dat (z progiem) — patrz: src/utils/helpers.py
try:
    from src.utils.helpers import smart_read, LoadMeta  # LoadMeta: dataclass z metadanymi
except Exception:  # kompat: gdy LoadMeta nie istnieje w starszej wersji
    from src.utils.helpers import smart_read  # type: ignore
    LoadMeta = object  # type: ignore

logger = logging.getLogger(__name__)

# === OVERLOADY DLA TYPE-CHECKERA ===
@overload
def load_dataframe(
    uploaded_file: Any,
    *,
    return_meta: Literal[False] = False,
    coerce_tz_to: Optional[str] = None,
    max_rows: Optional[int] = None,
) -> Optional[pd.DataFrame]: ...
@overload
def load_dataframe(
    uploaded_file: Any,
    *,
    return_meta: Literal[True],
    coerce_tz_to: Optional[str] = None,
    max_rows: Optional[int] = None,
) -> Tuple[pd.DataFrame, LoadMeta]: ...

# === API PUBLICZNE ===
def load_dataframe(
    uploaded_file: Any,
    *,
    return_meta: bool = False,
    coerce_tz_to: Optional[str] = None,
    max_rows: Optional[int] = None,
) -> Optional[pd.DataFrame] | Tuple[pd.DataFrame, LoadMeta]:
    """
    Uniwersalny loader z defensywną obsługą błędów.
    - return_meta=False (domyślnie): zachowuje stare API → zwraca tylko DataFrame.
    - return_meta=True: zwraca (DataFrame, LoadMeta).
    - coerce_tz_to: np. "UTC" — przekształca daty ze strefą do wskazanej i usuwa tz-info (naive) w celu spójności.
    - max_rows: opcjonalny limit wierszy po wczytaniu (np. do podglądu w UI).
    """
    if uploaded_file is None:
        return None

    try:
        # preferowana ścieżka — nowsza wersja helpers:
        try:
            df, meta = smart_read(uploaded_file, return_meta=True)  # type: ignore[arg-type]
        except TypeError:
            # kompatybilność ze starszą wersją smart_read
            df = smart_read(uploaded_file)  # type: ignore[assignment]
            meta = None  # type: ignore[assignment]

        # (opcjonalnie) ogranicz podgląd
        if isinstance(max_rows, int) and max_rows > 0 and len(df) > max_rows:
            df = df.head(max_rows).copy()

        # (opcjonalnie) ujednolić strefy czasowe → tz -> target -> naive
        if coerce_tz_to:
            for c in df.select_dtypes(include=["datetimetz"]).columns:
                try:
                    df[c] = df[c].dt.tz_convert(coerce_tz_to).dt.tz_localize(None)
                except Exception:
                    # jeśli kolumna jest tz-naive lub ma mieszane wartości
                    try:
                        df[c] = df[c].dt.tz_localize(coerce_tz_to).dt.tz_localize(None)
                    except Exception:
                        pass

        logger.info(
            "Loaded dataframe: rows=%s cols=%s meta=%s",
            len(df),
            len(df.columns),
            getattr(meta, "__dict__", None),
        )

        return (df, meta) if return_meta else df

    except Exception as e:
        # Spójny komunikat dla UI/FASTAPI
        raise RuntimeError(f"Error reading file: {e}") from e
