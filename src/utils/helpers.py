# src/utils/helpers.py
from __future__ import annotations
import re
from typing import Optional, Tuple

import numpy as np
import pandas as pd


# ============================
# SMART CAST NUMERIC (PRO)
# ============================

_CURRENCY_TOKENS = (
    "zł", "pln", "eur", "€", "$", "usd", "gbp", "£", "chf", "jpy", "¥", "aud", "cad", "nok", "sek", "czk",
)
_WS_CHARS = r"\u00A0\u202F"  # nbsp, narrow no-break space

_BOOL_TRUE = {"1", "true", "t", "yes", "y", "tak", "on"}
_BOOL_FALSE = {"0", "false", "f", "no", "n", "nie", "off"}


def _strip_currency_and_junk(s: pd.Series) -> pd.Series:
    """
    Usuwa spacje, separatory tysięcy, tokeny walutowe, kreski itp.
    Zachowuje tylko znaki numeryczne, kropki, przecinki, minus i nawiasy.
    """
    if not len(s):
        return s
    # normalizacja whitespace (także NBSP)
    s2 = s.astype(str).str.replace(f"[\\s{_WS_CHARS}]", "", regex=True, na=None)
    # ujemne w nawiasach: (123) -> -123
    s2 = s2.str.replace(r"^\((.*)\)$", r"-\1", regex=True, na=None)
    # usuń tokeny walutowe (case-insensitive)
    s2 = s2.str.replace("|".join([re.escape(t) for t in _CURRENCY_TOKENS]), "", case=False, regex=True, na=None)
    # procenty — oznacz wiersze, ale na razie usuń znak
    # (dzieleniem zajmiemy się po udanym parsowaniu)
    # UWAGA: zwrócimy też maskę procentów
    return s2


def _remove_thousands_and_fix_decimal(col: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """
    Heurystycznie usuwa separatory tysięcy i ujednolica przecinek/kropkę.
    Zwraca: (seryjka oczyszczona, maska_%_wartości)
    """
    s = _strip_currency_and_junk(col)

    # maska procentowa
    pct_mask = s.str.contains(r"%", na=False)
    s = s.str.replace("%", "", regex=False, na=None)

    # 1) podejście: europejskie ("1 234,56" / "1.234,56" / "1234,56")
    s_eu = s.str.replace(r"(?<=\d)[\.\s](?=\d{3}(\D|$))", "", regex=True, na=None)  # usuń kropki/spacje przed tysiącami
    s_eu = s_eu.str.replace(",", ".", regex=False, na=None)

    # 2) podejście: amerykańskie ("1,234.56" / "1234.56")
    s_us = s.str.replace(r"(?<=\d),(?=\d{3}(\D|$))", "", regex=True, na=None)      # usuń przecinki tysięcy
    # kropka już jest separatorem dziesiętnym

    # Spróbuj sparsować obie wersje i wybierz lepszą (mniej NaN)
    c_eu = pd.to_numeric(s_eu, errors="coerce")
    c_us = pd.to_numeric(s_us, errors="coerce")

    eu_ok = c_eu.notna().mean()
    us_ok = c_us.notna().mean()
    if eu_ok >= us_ok:
        return c_eu, pct_mask
    return c_us, pct_mask


def _try_parse_numeric(col: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """
    Wieloetapowe parsowanie: surowe -> usunięte śmieci -> EU/US -> brutalny fallback.
    Zwraca: (parsed_series, percent_mask)
    """
    s = col.astype(str)
    # najpierw szybkie podejście
    c0 = pd.to_numeric(s, errors="coerce")
    if c0.notna().mean() >= 0.9:
        return c0, s.str.contains("%", na=False)

    # czyszczenie i EU/US heurystyki
    c1, pct_mask = _remove_thousands_and_fix_decimal(s)
    if c1.notna().mean() >= 0.6:
        return c1, pct_mask

    # brutalny fallback: usuń wszystko poza cyfrą/kropką/minusem
    s2 = s.str.replace(f"[^{_WS_CHARS}0-9\\-\\.,eE+]", "", regex=True, na=None).str.replace(
        f"[\\s{_WS_CHARS}]", "", regex=True, na=None
    )
    # spróbuj jeszcze podejścia EU
    s2_eu = s2.str.replace(r"(?<=\d)\.(?=\d{3}(\D|$))", "", regex=True, na=None).str.replace(",", ".", regex=False, na=None)
    c2 = pd.to_numeric(s2_eu, errors="coerce")
    return c2, s.str.contains("%", na=False)


def _maybe_cast_boolean(col: pd.Series) -> Optional[pd.Series]:
    """Konwertuje znane formy bool (tak/nie, yes/no, 0/1, on/off) → Int8 (0/1)."""
    s = col.dropna().astype(str).str.strip().str.lower()
    if s.empty:
        return None
    uniq = set(s.unique())
    if uniq <= (_BOOL_TRUE | _BOOL_FALSE):
        mapped = col.astype(str).str.strip().str.lower().map(
            {**{k: 1 for k in _BOOL_TRUE}, **{k: 0 for k in _BOOL_FALSE}}
        )
        return mapped.astype("Int8")
    return None


def smart_cast_numeric(
    df: pd.DataFrame,
    max_unique_frac: float = 0.98,
    min_parse_ratio: float = 0.60,
) -> pd.DataFrame:
    """
    Inteligentne rzutowanie kolumn tekstowych na liczbowe.
    Obsługa:
      - formatów EU/US, walut, procentów, spacji NBSP, ujemnych w nawiasach,
      - booli tekstowych (tak/nie, yes/no, on/off → 0/1).
    Zasada akceptacji: rzutujemy kolumnę, jeśli:
      - >= min_parse_ratio wartości udało się sparsować, LUB
      - (po sparsowaniu) odsetek unikalnych <= max_unique_frac.
    """
    out = df.copy(deep=True)

    for c in list(out.columns):
        s = out[c]

        # najpierw bool tekstowy
        if s.dtype == "object":
            maybe_bool = _maybe_cast_boolean(s)
            if maybe_bool is not None:
                out[c] = maybe_bool
                continue

        if s.dtype == "object":
            parsed, pct_mask = _try_parse_numeric(s)

            parse_ratio = float(parsed.notna().mean()) if len(parsed) else 0.0
            unique_frac = float(parsed.nunique(dropna=True)) / max(1, len(parsed))

            if (parse_ratio >= min_parse_ratio) or (unique_frac <= max_unique_frac):
                # procenty -> przeskaluj do [0..1] tylko tam, gdzie wystąpił znak %
                if pct_mask.any():
                    parsed = parsed.where(~pct_mask, parsed / 100.0)
                out[c] = parsed

    # infinity → NaN (zostanie zaadresowane w cleanerze)
    out = out.replace([np.inf, -np.inf], np.nan)
    return out


# ============================
# INFER PROBLEM TYPE (PRO)
# ============================

_DATE_HINTS = ("date", "time", "timestamp", "data", "czas", "dt", "day", "month", "year")


def _has_datetime_signal(df: pd.DataFrame) -> bool:
    if isinstance(df.index, pd.DatetimeIndex):
        return True
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            return True
        # nazwa sugeruje datę?
        lc = str(col).lower()
        if any(h in lc for h in _DATE_HINTS):
            # szybka próba parsowania krótkiej próbki
            s = pd.to_datetime(df[col].astype(str).head(200), errors="coerce", infer_datetime_format=True)
            if s.notna().mean() > 0.6:
                return True
    return False


def _is_discrete_numeric(y: pd.Series, max_classes: int = 50, frac: float = 0.1) -> bool:
    """Czy numeryczna kolumna wygląda na etykiety (mała liczba wartości, zbliżone do całkowitych)."""
    if not pd.api.types.is_numeric_dtype(y):
        return False
    yv = y.dropna().values
    if len(yv) == 0:
        return False
    # blisko całkowitych?
    near_int = np.nanmean(np.abs(yv - np.round(yv)) <= 1e-9) > 0.99
    nuniq = int(pd.Series(yv).nunique())
    return near_int and (nuniq <= max(max_classes, int(frac * len(yv))))


def infer_problem_type(df: pd.DataFrame, target: str | None) -> str | None:
    """
    Heurystyka typu problemu:
      - 'timeseries' jeśli mamy sensowną oś czasu,
      - 'classification' jeśli target ma niską krotność (albo jest kategoryczny/tekstowy),
      - w przeciwnym razie 'regression'.
    """
    if target is None or target not in df.columns:
        return None

    y = df[target]

    # timeseries?
    if _has_datetime_signal(df):
        return "timeseries"

    # klasyfikacja?
    if (y.dtype == "object") or pd.api.types.is_categorical_dtype(y):
        return "classification"

    # numeryczny, ale dyskretny niczym etykiety
    if _is_discrete_numeric(y, max_classes=50, frac=0.05):
        return "classification"

    # binary float np. {0.0,1.0}
    yn = y.dropna().unique()
    if len(yn) <= 2 and all(v in (0, 1, 0.0, 1.0, True, False) for v in yn):
        return "classification"

    # fallback
    return "regression"


# ============================
# ENSURE DATETIME INDEX (PRO)
# ============================

def _best_parse_datetime(s: pd.Series) -> Optional[pd.Series]:
    """
    Próbuj parsować z dayfirst=True/False i wybierz wariant z mniejszą liczbą NaT.
    Zwraca serię datetime lub None.
    """
    if s is None or s.empty:
        return None
    s_str = s.astype(str)
    # usuń puste/placeholdery
    s_str = s_str.replace({"": np.nan, "None": np.nan, "NaN": np.nan})
    p1 = pd.to_datetime(s_str, errors="coerce", infer_datetime_format=True, dayfirst=False, utc=False)
    p2 = pd.to_datetime(s_str, errors="coerce", infer_datetime_format=True, dayfirst=True, utc=False)
    r1 = p1.notna().mean()
    r2 = p2.notna().mean()
    best = p1 if r1 >= r2 else p2
    if best.notna().mean() < 0.6:
        return None
    # ujednolić strefę: jeśli strefowa → sprowadź do naive (UTC-naive)
    try:
        if getattr(best.dt, "tz", None) is not None:
            best = best.dt.tz_convert("UTC").dt.tz_localize(None)
    except Exception:
        try:
            if getattr(best.dt, "tz", None) is not None:
                best = best.dt.tz_localize(None)
        except Exception:
            pass
    return best


def ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Znajduje i ustawia indeks czasowy:
      1) jeśli już jest DatetimeIndex → sortuje i zwraca,
      2) jeśli istnieje kolumna datetime → używa jej,
      3) heurystyka po nazwach kolumn + próba parsowania próbki,
      4) próba na pierwszej kolumnie (ostatnia deska ratunku).
    Nie modyfikuje merytorycznie danych (poza ustawieniem indeksu i sortowaniem).
    """
    if isinstance(df.index, pd.DatetimeIndex):
        return df.sort_index()

    out = df.copy()

    # 1) kolumny już datetime
    for c in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[c]):
            idx = out[c]
            # jeśli dużo NaT → pomiń
            if pd.isna(idx).mean() <= 0.4:
                out = out.set_index(c).sort_index()
                return out

    # 2) heurystyka po nazwie + parsowanie
    for c in out.columns:
        lc = str(c).lower()
        if any(h in lc for h in _DATE_HINTS):
            parsed = _best_parse_datetime(out[c])
            if parsed is not None:
                out = out.drop(columns=[c]).assign(_dt_idx=parsed).set_index("_dt_idx").sort_index()
                return out

    # 3) próba na pierwszej kolumnie
    if len(out.columns) > 0:
        first = out.columns[0]
        parsed = _best_parse_datetime(out[first])
        if parsed is not None:
            out = out.drop(columns=[first]).assign(_dt_idx=parsed).set_index("_dt_idx").sort_index()
            return out

    # 4) nie udało się
    return df
