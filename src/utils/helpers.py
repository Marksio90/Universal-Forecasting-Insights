"""
Utility Helpers - Zaawansowane funkcje pomocnicze dla przetwarzania danych.

Funkcjonalności:
- Smart numeric casting z obsługą walut, formatów EU/US, procentów
- Boolean text detection (yes/no, tak/nie, on/off)
- Problem type inference (classification, regression, timeseries)
- Datetime index detection i setup
- Robust parsing z multiple strategies
- Currency symbols handling
- NBSP and special whitespace handling
"""

from __future__ import annotations

import re
import logging
from typing import Optional, Tuple, Set, Dict, Any

import numpy as np
import pandas as pd

# ========================================================================================
# KONFIGURACJA
# ========================================================================================

# Currency tokens (rozszerzony set)
CURRENCY_SYMBOLS = (
    # Znaki
    "$", "€", "£", "¥", "₹", "₽", "₴", "₪", "₩", "₦", "₨", "₱", "฿", "₫", "₡", "₵",
    # Kody
    "zł", "pln", "eur", "usd", "gbp", "chf", "jpy", "aud", "cad", "nok", "sek", 
    "dkk", "czk", "huf", "ron", "bgn", "hrk", "rub", "try", "inr", "cny", "krw",
    "brl", "mxn", "zar", "sgd", "hkd", "nzd", "thb", "idr", "myr", "php", "vnd"
)

# Special whitespace characters
SPECIAL_WHITESPACE = r"\u00A0\u202F\u2009\u200A\u2002\u2003"  # NBSP, narrow, thin spaces

# Boolean mappings
BOOL_TRUE_VALUES: Set[str] = {"1", "true", "t", "yes", "y", "tak", "prawda", "on", "enabled"}
BOOL_FALSE_VALUES: Set[str] = {"0", "false", "f", "no", "n", "nie", "fałsz", "off", "disabled"}

# Date column hints
DATE_COLUMN_HINTS = (
    "date", "time", "timestamp", "datetime", "data", "czas", "dt", 
    "day", "month", "year", "fecha", "datum", "periodo", "period"
)

# Problem type inference
MAX_CLASSIFICATION_CLASSES = 50
MAX_CLASSIFICATION_FRACTION = 0.05
MIN_DATETIME_PARSE_RATIO = 0.6
MIN_NUMERIC_PARSE_RATIO = 0.60
MAX_UNIQUE_FRACTION = 0.98

# Thresholds
INTEGER_TOLERANCE = 1e-9
PERCENTAGE_SIGN = "%"

# ========================================================================================
# LOGGING
# ========================================================================================

def get_logger(name: str = "helpers", level: int = logging.INFO) -> logging.Logger:
    """
    Konfiguruje i zwraca logger bez duplikatów handlerów.
    
    Args:
        name: Nazwa loggera
        level: Poziom logowania
        
    Returns:
        Skonfigurowany logger
    """
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        logger.setLevel(level)
        handler = logging.StreamHandler()
        handler.setLevel(level)
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False
    
    return logger


LOGGER = get_logger()


# ========================================================================================
# NUMERIC CASTING - PREPROCESSING
# ========================================================================================

def _strip_currency_and_whitespace(series: pd.Series) -> pd.Series:
    """
    Usuwa waluty, specjalne spacje i inne śmieci.
    
    Args:
        series: Serie do oczyszczenia
        
    Returns:
        Oczyszczona serie
    """
    if len(series) == 0:
        return series
    
    # Konwertuj do string
    cleaned = series.astype(str)
    
    # Usuń wszystkie typy whitespace (także NBSP)
    cleaned = cleaned.str.replace(
        f"[\\s{SPECIAL_WHITESPACE}]",
        "",
        regex=True
    )
    
    # Ujemne w nawiasach: (123) -> -123
    cleaned = cleaned.str.replace(
        r"^\((.*)\)$",
        r"-\1",
        regex=True
    )
    
    # Usuń symbole walut (case-insensitive)
    currency_pattern = "|".join([re.escape(sym) for sym in CURRENCY_SYMBOLS])
    cleaned = cleaned.str.replace(
        currency_pattern,
        "",
        case=False,
        regex=True
    )
    
    return cleaned


def _detect_and_handle_percentages(series: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """
    Wykrywa i usuwa znak procentu, zwraca maskę.
    
    Args:
        series: Serie do przetworzenia
        
    Returns:
        Tuple (cleaned_series, percentage_mask)
    """
    # Maska wierszy z procentem
    percentage_mask = series.str.contains(PERCENTAGE_SIGN, na=False)
    
    # Usuń znak procentu
    cleaned = series.str.replace(PERCENTAGE_SIGN, "", regex=False)
    
    return cleaned, percentage_mask


def _parse_european_format(series: pd.Series) -> pd.Series:
    """
    Parsuje format europejski: 1.234,56 lub 1 234,56
    
    Args:
        series: Serie do parsowania
        
    Returns:
        Parsed numeric series
    """
    # Usuń kropki i spacje używane jako separatory tysięcy
    # Pattern: kropka/spacja przed dokładnie 3 cyframi
    cleaned = series.str.replace(
        r"(?<=\d)[\.\\s](?=\d{3}(\D|$))",
        "",
        regex=True
    )
    
    # Zamień przecinek na kropkę (separator dziesiętny)
    cleaned = cleaned.str.replace(",", ".", regex=False)
    
    # Parse
    return pd.to_numeric(cleaned, errors="coerce")


def _parse_american_format(series: pd.Series) -> pd.Series:
    """
    Parsuje format amerykański: 1,234.56
    
    Args:
        series: Serie do parsowania
        
    Returns:
        Parsed numeric series
    """
    # Usuń przecinki używane jako separatory tysięcy
    cleaned = series.str.replace(
        r"(?<=\d),(?=\d{3}(\D|$))",
        "",
        regex=True
    )
    
    # Kropka już jest separatorem dziesiętnym
    return pd.to_numeric(cleaned, errors="coerce")


def _aggressive_numeric_parse(series: pd.Series) -> pd.Series:
    """
    Agresywne parsowanie - usuwa wszystko oprócz cyfr, kropki, przecinka, minusa.
    
    Args:
        series: Serie do parsowania
        
    Returns:
        Parsed numeric series
    """
    # Zachowaj tylko: cyfry, kropka, przecinek, minus, e/E (notacja naukowa)
    cleaned = series.str.replace(
        f"[^{SPECIAL_WHITESPACE}0-9\\-\\.,eE+]",
        "",
        regex=True
    )
    
    # Usuń whitespace
    cleaned = cleaned.str.replace(
        f"[\\s{SPECIAL_WHITESPACE}]",
        "",
        regex=True
    )
    
    # Próbuj format europejski (bardziej popularny dla "zaszumionych" danych)
    cleaned = cleaned.str.replace(
        r"(?<=\d)\.(?=\d{3}(\D|$))",
        "",
        regex=True
    )
    cleaned = cleaned.str.replace(",", ".", regex=False)
    
    return pd.to_numeric(cleaned, errors="coerce")


def _try_parse_numeric_multistrategy(series: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """
    Multi-strategy parsing z wyborem najlepszego wyniku.
    
    Args:
        series: Serie do parsowania
        
    Returns:
        Tuple (best_parsed, percentage_mask)
    """
    series_str = series.astype(str)
    
    # Quick attempt - może już jest OK
    quick_parse = pd.to_numeric(series_str, errors="coerce")
    if quick_parse.notna().mean() >= 0.9:
        pct_mask = series_str.str.contains(PERCENTAGE_SIGN, na=False)
        return quick_parse, pct_mask
    
    # Preprocessing
    cleaned = _strip_currency_and_whitespace(series_str)
    cleaned, pct_mask = _detect_and_handle_percentages(cleaned)
    
    # Strategy 1: European format
    parsed_eu = _parse_european_format(cleaned)
    success_eu = parsed_eu.notna().mean()
    
    # Strategy 2: American format
    parsed_us = _parse_american_format(cleaned)
    success_us = parsed_us.notna().mean()
    
    # Wybierz lepszy
    if success_eu >= success_us and success_eu >= MIN_NUMERIC_PARSE_RATIO:
        LOGGER.debug(f"Used European format: {success_eu:.1%} success")
        return parsed_eu, pct_mask
    
    if success_us >= MIN_NUMERIC_PARSE_RATIO:
        LOGGER.debug(f"Used American format: {success_us:.1%} success")
        return parsed_us, pct_mask
    
    # Strategy 3: Aggressive
    parsed_aggressive = _aggressive_numeric_parse(series_str)
    success_aggressive = parsed_aggressive.notna().mean()
    
    LOGGER.debug(f"Used aggressive parsing: {success_aggressive:.1%} success")
    return parsed_aggressive, pct_mask


# ========================================================================================
# BOOLEAN DETECTION
# ========================================================================================

def _detect_and_cast_boolean(series: pd.Series) -> Optional[pd.Series]:
    """
    Wykrywa i konwertuje tekstowe wartości boolean.
    
    Args:
        series: Serie do sprawdzenia
        
    Returns:
        Int8 series (0/1) lub None jeśli nie jest boolean
    """
    # Drop NaN dla analizy
    clean = series.dropna().astype(str).str.strip().str.lower()
    
    if clean.empty:
        return None
    
    unique_values = set(clean.unique())
    
    # Sprawdź czy wszystkie wartości są w known boolean values
    if unique_values <= (BOOL_TRUE_VALUES | BOOL_FALSE_VALUES):
        # Mapping
        bool_map = {
            **{val: 1 for val in BOOL_TRUE_VALUES},
            **{val: 0 for val in BOOL_FALSE_VALUES}
        }
        
        mapped = series.astype(str).str.strip().str.lower().map(bool_map)
        
        LOGGER.debug(f"Detected boolean column with values: {unique_values}")
        return mapped.astype("Int8")
    
    return None


# ========================================================================================
# SMART NUMERIC CASTING (MAIN)
# ========================================================================================

def smart_cast_numeric(
    df: pd.DataFrame,
    max_unique_frac: float = MAX_UNIQUE_FRACTION,
    min_parse_ratio: float = MIN_NUMERIC_PARSE_RATIO,
    convert_percentages: bool = True
) -> pd.DataFrame:
    """
    Inteligentne konwertowanie kolumn do numeric.
    
    Obsługuje:
    - Formaty EU i US (separatory tysięcy i dziesiętne)
    - Symbole walut (zł, $, €, £, ¥, etc.)
    - Procenty (konwersja do [0, 1])
    - Ujemne w nawiasach (123) -> -123
    - Specjalne whitespace (NBSP, narrow spaces)
    - Boolean tekstowy (yes/no, tak/nie, on/off -> 0/1)
    
    Zasada akceptacji:
    - Rzutujemy jeśli >= min_parse_ratio wartości się sparsowało
    - LUB unique fraction <= max_unique_frac (prawdopodobnie kategorie liczbowe)
    
    Args:
        df: DataFrame do przetworzenia
        max_unique_frac: Max frakcja unikalnych wartości (default: 0.98)
        min_parse_ratio: Min frakcja poprawnie sparsowanych (default: 0.60)
        convert_percentages: Czy konwertować % do [0,1] (default: True)
        
    Returns:
        DataFrame z przekonwertowanymi kolumnami
        
    Example:
        >>> df = pd.DataFrame({
        ...     "price": ["$1,234.56", "$2,345.67"],
        ...     "tax": ["15%", "20%"],
        ...     "active": ["yes", "no"]
        ... })
        >>> df_clean = smart_cast_numeric(df)
        >>> print(df_clean.dtypes)
    """
    result = df.copy(deep=True)
    conversions = 0
    
    for col in result.columns:
        series = result[col]
        
        # Skip już numeric
        if pd.api.types.is_numeric_dtype(series):
            continue
        
        # Skip jeśli nie object (np. datetime)
        if series.dtype != "object":
            continue
        
        # 1. Próba boolean
        bool_series = _detect_and_cast_boolean(series)
        if bool_series is not None:
            result[col] = bool_series
            conversions += 1
            LOGGER.debug(f"Converted '{col}' to boolean (Int8)")
            continue
        
        # 2. Próba numeric
        parsed, pct_mask = _try_parse_numeric_multistrategy(series)
        
        # Sprawdź success rate
        parse_ratio = parsed.notna().mean()
        unique_frac = parsed.nunique(dropna=True) / max(1, len(parsed))
        
        accept = (parse_ratio >= min_parse_ratio) or (unique_frac <= max_unique_frac)
        
        if accept:
            # Konwersja procentów
            if convert_percentages and pct_mask.any():
                n_pct = pct_mask.sum()
                parsed = parsed.where(~pct_mask, parsed / 100.0)
                LOGGER.debug(f"Converted {n_pct} percentage values in '{col}'")
            
            result[col] = parsed
            conversions += 1
            LOGGER.debug(
                f"Converted '{col}' to numeric "
                f"(parse_ratio={parse_ratio:.1%}, unique_frac={unique_frac:.1%})"
            )
    
    # Cleanup infinities
    result = result.replace([np.inf, -np.inf], np.nan)
    
    if conversions > 0:
        LOGGER.info(f"Smart cast: converted {conversions} columns to numeric/boolean")
    
    return result


# ========================================================================================
# PROBLEM TYPE INFERENCE
# ========================================================================================

def _has_datetime_signal(df: pd.DataFrame) -> bool:
    """
    Sprawdza czy DataFrame ma sygnał czasowy.
    
    Args:
        df: DataFrame do sprawdzenia
        
    Returns:
        True jeśli wykryto datetime
    """
    # Check index
    if isinstance(df.index, pd.DatetimeIndex):
        return True
    
    # Check columns
    for col in df.columns:
        # Already datetime type
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            return True
        
        # Name suggests datetime
        col_lower = str(col).lower()
        if any(hint in col_lower for hint in DATE_COLUMN_HINTS):
            # Quick parse test on sample
            sample = df[col].astype(str).head(200)
            parsed = pd.to_datetime(sample, errors="coerce")
            
            if parsed.notna().mean() > MIN_DATETIME_PARSE_RATIO:
                return True
    
    return False


def _is_discrete_numeric(
    series: pd.Series,
    max_classes: int = MAX_CLASSIFICATION_CLASSES,
    max_frac: float = MAX_CLASSIFICATION_FRACTION
) -> bool:
    """
    Sprawdza czy numeryczna seria wygląda na kategoryczną.
    
    Args:
        series: Serie do sprawdzenia
        max_classes: Max liczba unikalnych wartości
        max_frac: Max frakcja unikalnych wartości
        
    Returns:
        True jeśli wygląda na kategoryczną
    """
    if not pd.api.types.is_numeric_dtype(series):
        return False
    
    values = series.dropna().values
    
    if len(values) == 0:
        return False
    
    # Sprawdź czy blisko integer
    is_near_integer = np.mean(np.abs(values - np.round(values)) <= INTEGER_TOLERANCE) > 0.99
    
    # Sprawdź liczbę unikalnych
    n_unique = int(pd.Series(values).nunique())
    
    # Accept jako discrete jeśli:
    # - są prawie integer AND
    # - mało unikalnych wartości (absolute lub relative)
    threshold = max(max_classes, int(max_frac * len(values)))
    
    return is_near_integer and n_unique <= threshold


def infer_problem_type(
    df: pd.DataFrame,
    target: Optional[str]
) -> Optional[str]:
    """
    Wykrywa typ problemu ML na podstawie danych.
    
    Logika:
    1. Timeseries - jeśli jest datetime index lub kolumny
    2. Classification - jeśli target jest:
       - Object/categorical
       - Discrete numeric (mało wartości, integer-like)
       - Binary (0/1, True/False)
    3. Regression - w pozostałych przypadkach
    
    Args:
        df: DataFrame z danymi
        target: Nazwa kolumny celu (optional)
        
    Returns:
        "timeseries", "classification", "regression" lub None
        
    Example:
        >>> df = pd.DataFrame({
        ...     "date": pd.date_range("2020", periods=100),
        ...     "value": np.random.randn(100)
        ... })
        >>> problem_type = infer_problem_type(df, "value")
        >>> print(problem_type)  # "timeseries"
    """
    # Brak targetu
    if target is None or target not in df.columns:
        LOGGER.warning("Target not specified or not found in DataFrame")
        return None
    
    target_series = df[target]
    
    # 1. Timeseries detection
    if _has_datetime_signal(df):
        LOGGER.debug("Detected problem type: timeseries (datetime signal found)")
        return "timeseries"
    
    # 2. Classification detection
    
    # Object or categorical dtype
    if target_series.dtype == "object" or pd.api.types.is_categorical_dtype(target_series):
        LOGGER.debug("Detected problem type: classification (object/categorical dtype)")
        return "classification"
    
    # Discrete numeric (looks like labels)
    if _is_discrete_numeric(target_series):
        n_unique = target_series.nunique(dropna=True)
        LOGGER.debug(f"Detected problem type: classification (discrete numeric, {n_unique} classes)")
        return "classification"
    
    # Binary float (0.0/1.0)
    unique_values = target_series.dropna().unique()
    if len(unique_values) <= 2:
        is_binary = all(
            val in (0, 1, 0.0, 1.0, True, False)
            for val in unique_values
        )
        if is_binary:
            LOGGER.debug("Detected problem type: classification (binary)")
            return "classification"
    
    # 3. Default: regression
    LOGGER.debug("Detected problem type: regression (default)")
    return "regression"


# ========================================================================================
# DATETIME INDEX SETUP
# ========================================================================================

def _parse_datetime_dual_strategy(series: pd.Series) -> Optional[pd.Series]:
    """
    Parsuje datetime z dual strategy (dayfirst=True/False).
    
    Args:
        series: Serie do parsowania
        
    Returns:
        Parsed datetime series lub None
    """
    if series is None or series.empty:
        return None
    
    # Konwertuj do string i cleanup
    series_str = series.astype(str)
    series_str = series_str.replace({
        "": np.nan,
        "None": np.nan,
        "NaN": np.nan,
        "nan": np.nan,
        "NaT": np.nan
    })
    
    # Strategy 1: dayfirst=False (American: MM/DD/YYYY)
    parsed_us = pd.to_datetime(
        series_str,
        errors="coerce",
        infer_datetime_format=True,
        dayfirst=False
    )
    success_us = parsed_us.notna().mean()
    
    # Strategy 2: dayfirst=True (European: DD/MM/YYYY)
    parsed_eu = pd.to_datetime(
        series_str,
        errors="coerce",
        infer_datetime_format=True,
        dayfirst=True
    )
    success_eu = parsed_eu.notna().mean()
    
    # Wybierz lepszy
    best_parsed = parsed_us if success_us >= success_eu else parsed_eu
    best_success = max(success_us, success_eu)
    
    # Minimum threshold
    if best_success < MIN_DATETIME_PARSE_RATIO:
        return None
    
    # Normalize timezone
    try:
        if hasattr(best_parsed.dt, "tz") and best_parsed.dt.tz is not None:
            # Convert to UTC and remove timezone
            best_parsed = best_parsed.dt.tz_convert("UTC").dt.tz_localize(None)
    except Exception:
        # Fallback: just remove timezone
        try:
            if hasattr(best_parsed.dt, "tz") and best_parsed.dt.tz is not None:
                best_parsed = best_parsed.dt.tz_localize(None)
        except Exception:
            pass
    
    LOGGER.debug(f"Parsed datetime: {best_success:.1%} success")
    return best_parsed


def ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Wykrywa i ustawia datetime index.
    
    Strategia:
    1. Jeśli już jest DatetimeIndex - sortuj i zwróć
    2. Jeśli istnieje kolumna datetime - użyj jej
    3. Heurystyka po nazwach + parsing
    4. Próba na pierwszej kolumnie (last resort)
    
    Args:
        df: DataFrame do przetworzenia
        
    Returns:
        DataFrame z DatetimeIndex (jeśli się udało)
        
    Example:
        >>> df = pd.DataFrame({
        ...     "date": ["2020-01-01", "2020-01-02"],
        ...     "value": [1, 2]
        ... })
        >>> df_indexed = ensure_datetime_index(df)
        >>> isinstance(df_indexed.index, pd.DatetimeIndex)
        True
    """
    # Already has DatetimeIndex
    if isinstance(df.index, pd.DatetimeIndex):
        LOGGER.debug("DataFrame already has DatetimeIndex")
        return df.sort_index()
    
    result = df.copy()
    
    # 1. Check for existing datetime columns
    for col in result.columns:
        if pd.api.types.is_datetime64_any_dtype(result[col]):
            # Check quality (not too many NaT)
            if pd.isna(result[col]).mean() <= 0.4:
                result = result.set_index(col).sort_index()
                LOGGER.debug(f"Set DatetimeIndex from existing datetime column: '{col}'")
                return result
    
    # 2. Heuristic by column name + parsing
    for col in result.columns:
        col_lower = str(col).lower()
        
        if any(hint in col_lower for hint in DATE_COLUMN_HINTS):
            parsed = _parse_datetime_dual_strategy(result[col])
            
            if parsed is not None:
                result = result.drop(columns=[col])
                result["_dt_index"] = parsed
                result = result.set_index("_dt_index").sort_index()
                LOGGER.debug(f"Set DatetimeIndex from column: '{col}' (by hint)")
                return result
    
    # 3. Try first column (last resort)
    if len(result.columns) > 0:
        first_col = result.columns[0]
        parsed = _parse_datetime_dual_strategy(result[first_col])
        
        if parsed is not None:
            result = result.drop(columns=[first_col])
            result["_dt_index"] = parsed
            result = result.set_index("_dt_index").sort_index()
            LOGGER.debug(f"Set DatetimeIndex from first column: '{first_col}'")
            return result
    
    # Failed to find datetime
    LOGGER.debug("Could not find or parse datetime column")
    return df


# ========================================================================================
# ADDITIONAL UTILITIES
# ========================================================================================

def detect_encoding(file_path: str) -> str:
    """
    Wykrywa encoding pliku (helper dla file parsers).
    
    Args:
        file_path: Ścieżka do pliku
        
    Returns:
        Wykryty encoding (default: 'utf-8')
    """
    try:
        import chardet
        
        with open(file_path, 'rb') as f:
            raw_data = f.read(100000)  # Read first 100KB
            result = chardet.detect(raw_data)
            encoding = result['encoding']
            confidence = result['confidence']
            
            if confidence > 0.7:
                LOGGER.debug(f"Detected encoding: {encoding} (confidence: {confidence:.1%})")
                return encoding
    except ImportError:
        LOGGER.debug("chardet not available, using utf-8")
    except Exception as e:
        LOGGER.warning(f"Encoding detection failed: {e}")
    
    return 'utf-8'


def safe_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizuje nazwy kolumn (usuwa spacje, spec. znaki).
    
    Args:
        df: DataFrame
        
    Returns:
        DataFrame z znormalizowanymi nazwami
    """
    result = df.copy()
    
    new_names = {}
    for col in result.columns:
        # Convert to string
        col_str = str(col)
        
        # Replace spaces with underscore
        normalized = col_str.strip().replace(" ", "_")
        
        # Remove special characters (keep alphanumeric, underscore, dash)
        normalized = re.sub(r'[^a-zA-Z0-9_\-]', '', normalized)
        
        # Lowercase
        normalized = normalized.lower()
        
        # Remove leading/trailing underscores
        normalized = normalized.strip("_")
        
        if normalized != col_str:
            new_names[col] = normalized
    
    if new_names:
        result = result.rename(columns=new_names)
        LOGGER.debug(f"Normalized {len(new_names)} column names")
    
    return result