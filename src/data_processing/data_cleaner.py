from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Iterable, Tuple, Dict, Any
import re
import numpy as np
import pandas as pd

__all__ = ["CleanConfig", "clean_dataframe"]

@dataclass
class CleanConfig:
    # Nazwy/teksty
    strip_unicode: bool = True                 # usuń nietypowe spacje (NBSP itp.)
    normalize_whitespace: bool = True          # zredukuj wielokrotne spacje do jednej
    lowercase_categoricals: bool = False       # znormalizuj case w kolumnach tekstowych
    standardize_colnames: bool = True          # przytnij i zamień spacje w nazwach kolumn na _
    # Braki
    missing_tokens: Tuple[str, ...] = ("", "na", "n/a", "none", "null", "nan", "-", "--")
    # Konwersje
    coerce_numeric_from_str: bool = True       # spróbuj konwersji tekst→liczba (obsługa ',' i separatorów tysięcy)
    parse_dates: bool = True                   # spróbuj wykryć i sparsować daty
    date_infer: bool = True                    # infer_datetime_format
    date_dayfirst: bool = False                # np. PL: True, jeśli dane w formacie dd-mm-rrrr
    min_parse_success_ratio: float = 0.8       # próg sukcesów, aby zatwierdzić konwersję kolumny
    # Naprawy liczb
    replace_inf: bool = True
    impute_numeric: str = "median"             # median|mean|constant|none
    impute_numeric_constant: float = 0.0
    clip_quantiles: Optional[Tuple[float, float]] = None  # np. (0.01, 0.99) aby przyciąć outliery
    # Tekst
    impute_categorical_value: str = "missing"
    # Typy
    cast_low_card_to_category: bool = True
    max_unique_for_category: int = 1000
    unique_fraction_for_category: float = 0.5  # jeśli unikalnych <= 50% wierszy → kategoria
    # Duplikaty
    drop_duplicates: bool = True
    duplicates_subset: Optional[Iterable[str]] = None

_WS_RE = re.compile(r"\s+", flags=re.UNICODE)
_UNICODE_SPACES = re.compile(r"[\u00A0\u1680\u2000-\u200A\u202F\u205F\u3000]")  # NBSP i pokrewne

def _normalize_colnames(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [
        _WS_RE.sub("_", c.strip()) if isinstance(c, str) else c
        for c in df.columns
    ]
    return df

def _clean_text_series(s: pd.Series, cfg: CleanConfig) -> pd.Series:
    s = s.astype("string")
    if cfg.strip_unicode:
        s = s.str.replace(_UNICODE_SPACES, " ", regex=True)
    if cfg.normalize_whitespace:
        s = s.str.replace(_WS_RE, " ", regex=True)
    s = s.str.strip()
    # Missing tokens → <NA>
    toks = set(t.lower() for t in cfg.missing_tokens)
    s = s.map(lambda x: pd.NA if (x is None or (isinstance(x, str) and x.lower() in toks)) else x)
    if cfg.lowercase_categoricals:
        s = s.str.lower()
    return s

def _try_coerce_numeric(col: pd.Series, cfg: CleanConfig) -> pd.Series:
    # Zamień popularne formaty: "(1,234.5)" → "-1234.5" ; "1 234,56" → "1234.56"
    txt = col.astype("string")
    # nawiasy na minus
    txt = txt.str.replace(r"^\((.*)\)$", r"-\1", regex=True)
    # usuń spacje, kropki i przecinki tysięcy (zachowując separator dziesiętny)
    # heurystyka: najpierw zamień przecinek na kropkę, potem usuń spacje i apostrofy
    txt = txt.str.replace(",", ".", regex=False)
    txt = txt.str.replace(r"[ '\u00A0]", "", regex=True)
    # jeśli zostały separatory tysięcy jako kropki, usuń je, gdy występują >1 punkt
    txt = txt.str.replace(r"(?<=\d)\.(?=\d{3}(\D|$))", "", regex=True)
    coerced = pd.to_numeric(txt, errors="coerce")
    success = coerced.notna().mean() if len(coerced) else 0.0
    return coerced if success >= cfg.min_parse_success_ratio else col

def _try_parse_dates(col: pd.Series, cfg: CleanConfig) -> pd.Series:
    txt = col.astype("string")
    parsed = pd.to_datetime(
        txt,
        errors="coerce",
        infer_datetime_format=cfg.date_infer,
        dayfirst=cfg.date_dayfirst,
        utc=False,
    )
    success = parsed.notna().mean() if len(parsed) else 0.0
    return parsed if success >= cfg.min_parse_success_ratio else col

def _impute_numeric_block(df: pd.DataFrame, cfg: CleanConfig) -> pd.DataFrame:
    num = df.select_dtypes(include=[np.number]).columns
    if len(num) == 0:
        return df
    if cfg.replace_inf:
        df[num] = df[num].replace([np.inf, -np.inf], np.nan)
    if cfg.impute_numeric == "median":
        fill = {c: df[c].median() for c in num}
        df[num] = df[num].fillna(fill)
    elif cfg.impute_numeric == "mean":
        fill = {c: df[c].mean() for c in num}
        df[num] = df[num].fillna(fill)
    elif cfg.impute_numeric == "constant":
        df[num] = df[num].fillna(cfg.impute_numeric_constant)
    # opcjonalne przycięcie outlierów
    if cfg.clip_quantiles:
        ql, qh = cfg.clip_quantiles
        q = df[num].quantile([ql, qh])
        lo, hi = q.loc[ql], q.loc[qh]
        df[num] = df[num].clip(lower=lo, upper=hi, axis=1)
    return df

def _impute_categoricals(df: pd.DataFrame, cfg: CleanConfig) -> pd.DataFrame:
    obj = df.select_dtypes(include=["object", "string"]).columns
    if len(obj) == 0:
        return df
    df[obj] = df[obj].fillna(cfg.impute_categorical_value)
    return df

def _cast_low_card_to_category(df: pd.DataFrame, cfg: CleanConfig) -> pd.DataFrame:
    if not cfg.cast_low_card_to_category:
        return df
    cols = df.select_dtypes(include=["object", "string"]).columns
    n = len(df)
    for c in cols:
        nun = df[c].nunique(dropna=False)
        if nun <= cfg.max_unique_for_category and nun / max(1, n) <= cfg.unique_fraction_for_category:
            df[c] = df[c].astype("category")
    return df

def clean_dataframe(df: pd.DataFrame, config: Optional[CleanConfig] = None) -> pd.DataFrame:
    """
    PRO+++ czyszczenie DataFrame:
      - normalizacja nazw kolumn (opcjonalnie),
      - trimming/normalizacja białych znaków i tokenów braków,
      - bezpieczne konwersje: tekst→liczba, tekst→data (prog skuteczności),
      - naprawa inf/NaN i imputacja (median/mean/const),
      - opcjonalne przycięcie outlierów po kwantylach,
      - imputacja kategorii, opcjonalne rzutowanie na `category`,
      - usunięcie duplikatów.

    Wstecznie kompatybilne: `clean_dataframe(df)` działa jak dotychczas (z sensownymi domyślnymi).
    """
    cfg = config or CleanConfig()
    if not isinstance(df, pd.DataFrame):
        raise TypeError("clean_dataframe: expected pandas.DataFrame")

    out = df.copy()

    # 1) Nazwy kolumn
    if cfg.standardize_colnames:
        out = _normalize_colnames(out)

    # 2) Teksty: strip/normalize/missing tokens
    obj_cols = out.select_dtypes(include=["object", "string"]).columns.tolist()
    for c in obj_cols:
        out[c] = _clean_text_series(out[c], cfg)

    # 3) Konwersje: tekst→liczba / tekst→data (bezpiecznie, gdy >= min_parse_success_ratio)
    if cfg.coerce_numeric_from_str and obj_cols:
        for c in obj_cols:
            # nie próbuj parsować daty jako liczby; najpierw data
            if cfg.parse_dates:
                parsed_dt = _try_parse_dates(out[c], cfg)
                if pd.api.types.is_datetime64_any_dtype(parsed_dt):
                    out[c] = parsed_dt
                    continue
            # liczby
            coerced_num = _try_coerce_numeric(out[c], cfg)
            if pd.api.types.is_float_dtype(coerced_num) or pd.api.types.is_integer_dtype(coerced_num):
                out[c] = coerced_num

    # 4) Daty dla pozostałych tekstów (jeśli jeszcze nie parsowane)
    if cfg.parse_dates:
        obj_cols = out.select_dtypes(include=["object", "string"]).columns.tolist()
        for c in obj_cols:
            maybe_dt = _try_parse_dates(out[c], cfg)
            if pd.api.types.is_datetime64_any_dtype(maybe_dt):
                out[c] = maybe_dt

    # 5) Naprawy liczb + imputacja
    out = _impute_numeric_block(out, cfg)

    # 6) Imputacja kategorii
    out = _impute_categoricals(out, cfg)

    # 7) Rzutowanie niskiej kardynalności na category
    out = _cast_low_card_to_category(out, cfg)

    # 8) Duplikaty
    if cfg.drop_duplicates:
        out = out.drop_duplicates(subset=list(cfg.duplicates_subset) if cfg.duplicates_subset else None, keep="first").reset_index(drop=True)

    return out
