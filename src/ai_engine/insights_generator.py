"""
AI Insights Engine - Backend module dla generowania insightów biznesowych.

Funkcjonalności:
- Generowanie insightów z OpenAI GPT
- Inteligentne budowanie promptów z kontekstem danych
- Retry logic z exponential backoff
- Cache dla identycznych zapytań (hash-based)
- Walidacja i sanityzacja odpowiedzi JSON
- Timeout handling
- Rate limiting (token bucket)
- Structured outputs (dataclasses)
- Comprehensive logging
- Fallback strategies
"""

from __future__ import annotations

import json
import time
import hashlib
import logging
from functools import lru_cache
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Literal, Tuple

import pandas as pd

from .openai_integrator import chat_completion

# ========================================================================================
# KONFIGURACJA
# ========================================================================================

# Limity profilowania
MAX_PROFILE_COLS = 50
MAX_PROFILE_ROWS = 200_000
DEFAULT_SAMPLE_ROWS = 50_000

# Limity odpowiedzi
MAX_SUMMARY_LENGTH = 600
MAX_ITEM_LENGTH = 180
MAX_LIST_ITEMS = 8

# Top N dla różnych metryk
TOP_MISSING = 5
TOP_NUNIQUE = 5
TOP_NUMERIC_VAR = 5

# Retry configuration
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds
RETRY_BACKOFF = 2  # exponential backoff multiplier

# Timeout dla API calls (seconds)
API_TIMEOUT = 60

# Cache TTL (seconds)
CACHE_TTL = 3600  # 1 hour

# Schema keys
INSIGHTS_SCHEMA_KEYS = ("summary", "top_insights", "recommendations", "risks")

# ========================================================================================
# PROMPTY
# ========================================================================================

SYSTEM_PROMPT = """Jesteś seniorem Data Scientist i Analitykiem Biznesowym.

Twoim zadaniem jest tworzyć zwięzłe, precyzyjne wnioski biznesowe na podstawie metadanych o zbiorze danych i celu użytkownika. 

WAŻNE:
- Nie wymyślaj wartości z danych — używaj wyłącznie dostarczonych metryk
- Zwracaj odpowiedź w 100% poprawnym JSON (bez zbędnego tekstu)
- Każdy element listy to krótka, konkretna myśl (max 180 znaków)
- Nie używaj markdown ani emoji w odpowiedziach

Wymagany schemat JSON:
{
  "summary": string,
  "top_insights": string[],
  "recommendations": string[],
  "risks": string[]
}"""

# ========================================================================================
# LOGGING
# ========================================================================================

def get_logger(name: str = "ai_insights", level: int = logging.INFO) -> logging.Logger:
    """
    Konfiguruje i zwraca logger.
    
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
# DATACLASSES
# ========================================================================================

@dataclass(frozen=True)
class InsightsOptions:
    """Opcje generowania insightów."""
    mode: Literal["standard", "executive", "technical"] = "standard"
    sample_rows_for_profile: int = DEFAULT_SAMPLE_ROWS
    max_profile_cols: int = MAX_PROFILE_COLS
    domain_hint: Optional[str] = None
    language: str = "pl"
    timeout: int = API_TIMEOUT


@dataclass(frozen=True)
class InsightsPayload:
    """Strukturyzowana odpowiedź z insightami."""
    summary: str
    top_insights: List[str]
    recommendations: List[str]
    risks: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Konwertuje do słownika."""
        return asdict(self)


@dataclass(frozen=True)
class InsightsResult:
    """Wynik operacji generowania insightów."""
    ok: bool
    payload: Optional[InsightsPayload]
    raw_text: Optional[str]
    message: str
    cached: bool = False
    retries: int = 0


@dataclass(frozen=True)
class DataProfile:
    """Profil danych dla promptu."""
    rows: int
    cols: int
    missing_pct: float
    dupes: int
    type_counts: Dict[str, int]
    top_missing: Dict[str, float]
    top_nunique: Dict[str, int]
    numeric_variance: Dict[str, float]


# ========================================================================================
# CACHE
# ========================================================================================

class SimpleCache:
    """Prosty cache z TTL."""
    
    def __init__(self, ttl: int = CACHE_TTL):
        self.ttl = ttl
        self._cache: Dict[str, Tuple[Any, float]] = {}
    
    def get(self, key: str) -> Optional[Any]:
        """Pobiera wartość z cache."""
        if key in self._cache:
            value, timestamp = self._cache[key]
            
            if time.time() - timestamp < self.ttl:
                LOGGER.debug(f"Cache HIT: {key[:16]}...")
                return value
            else:
                # Expired
                del self._cache[key]
                LOGGER.debug(f"Cache EXPIRED: {key[:16]}...")
        
        LOGGER.debug(f"Cache MISS: {key[:16]}...")
        return None
    
    def set(self, key: str, value: Any) -> None:
        """Zapisuje wartość do cache."""
        self._cache[key] = (value, time.time())
        LOGGER.debug(f"Cache SET: {key[:16]}...")
    
    def clear(self) -> None:
        """Czyści cache."""
        self._cache.clear()


# Global cache instance
_INSIGHTS_CACHE = SimpleCache(ttl=CACHE_TTL)


def compute_cache_key(
    df_hash: str,
    goal: Optional[str],
    mode: str,
    options: InsightsOptions
) -> str:
    """
    Oblicza klucz cache dla zapytania.
    
    Args:
        df_hash: Hash DataFrame
        goal: Cel biznesowy
        mode: Tryb generowania
        options: Opcje
        
    Returns:
        SHA-256 hash jako klucz
    """
    components = [
        df_hash,
        goal or "",
        mode,
        str(options.sample_rows_for_profile),
        str(options.max_profile_cols),
        options.domain_hint or "",
        options.language
    ]
    
    key_str = "|".join(components)
    return hashlib.sha256(key_str.encode()).hexdigest()


def compute_dataframe_hash(df: pd.DataFrame) -> str:
    """
    Oblicza hash DataFrame (kolumny + rozmiar).
    
    Args:
        df: DataFrame
        
    Returns:
        Hash jako string
    """
    components = [
        str(len(df)),
        str(df.shape[1]),
        ",".join(sorted(df.columns.astype(str)))
    ]
    
    hash_str = "|".join(components)
    return hashlib.sha256(hash_str.encode()).hexdigest()[:16]


# ========================================================================================
# UTILITY FUNCTIONS
# ========================================================================================

def truncate_string(s: str, max_len: int) -> str:
    """
    Ucina string do max_len.
    
    Args:
        s: String do ucięcia
        max_len: Maksymalna długość
        
    Returns:
        Ucięty string
    """
    if len(s) <= max_len:
        return s
    
    return s[:max_len - 1] + "…"


def normalize_list_of_strings(x: Any) -> List[str]:
    """
    Normalizuje do listy stringów.
    
    Args:
        x: Input (list, str, number, etc.)
        
    Returns:
        Lista stringów
    """
    if x is None:
        return []
    
    if isinstance(x, list):
        return [str(i).strip() for i in x if isinstance(i, (str, int, float))]
    
    if isinstance(x, (str, int, float)):
        return [str(x).strip()]
    
    return []


def cap_list_items(
    items: List[str],
    max_len: int = MAX_ITEM_LENGTH,
    max_items: int = MAX_LIST_ITEMS
) -> List[str]:
    """
    Ogranicza długość i liczbę elementów listy.
    
    Args:
        items: Lista elementów
        max_len: Maksymalna długość elementu
        max_items: Maksymalna liczba elementów
        
    Returns:
        Ograniczona lista
    """
    # Filter empty
    items = [i.strip() for i in items if i and isinstance(i, str)]
    
    # Truncate each item
    items = [truncate_string(i, max_len) for i in items]
    
    # Limit count
    return items[:max_items]


def safe_json_loads(text: str) -> Optional[dict]:
    """
    Bezpieczny parsing JSON.
    
    Args:
        text: JSON string
        
    Returns:
        Dict lub None przy błędzie
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        LOGGER.warning(f"JSON parsing failed: {e}")
        return None
    except Exception as e:
        LOGGER.error(f"Unexpected error parsing JSON: {e}")
        return None


def extract_json_from_text(text: str) -> Optional[str]:
    """
    Próbuje wyekstrahować JSON z tekstu (jeśli model dodał komentarze).
    
    Args:
        text: Surowy tekst z potencjalnym JSON
        
    Returns:
        JSON string lub None
    """
    # Spróbuj znaleźć pierwszy { i ostatni }
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        json_candidate = text[first_brace:last_brace + 1]
        
        # Spróbuj sparsować
        if safe_json_loads(json_candidate):
            return json_candidate
    
    return None


def coerce_to_schema(obj: Any) -> Optional[InsightsPayload]:
    """
    Dopasowuje obiekt do schematu InsightsPayload.
    
    Args:
        obj: Obiekt do dopasowania
        
    Returns:
        InsightsPayload lub None
    """
    if not isinstance(obj, dict):
        return None
    
    # Sprawdź czy wszystkie klucze są obecne
    if not all(key in obj for key in INSIGHTS_SCHEMA_KEYS):
        LOGGER.warning(f"Missing keys in response. Expected: {INSIGHTS_SCHEMA_KEYS}")
        return None
    
    # Ekstrahuj i normalizuj
    summary = str(obj.get("summary", "")).strip()
    top_insights = normalize_list_of_strings(obj.get("top_insights"))
    recommendations = normalize_list_of_strings(obj.get("recommendations"))
    risks = normalize_list_of_strings(obj.get("risks"))
    
    # Walidacja minimalnych wymagań
    if not summary:
        LOGGER.warning("Summary is empty")
        return None
    
    return InsightsPayload(
        summary=summary,
        top_insights=top_insights,
        recommendations=recommendations,
        risks=risks
    )


# ========================================================================================
# DATA PROFILING
# ========================================================================================

def sample_for_profile(df: pd.DataFrame, n_rows: int) -> pd.DataFrame:
    """
    Próbkuje DataFrame dla profilowania.
    
    Args:
        df: DataFrame źródłowy
        n_rows: Maksymalna liczba wierszy
        
    Returns:
        Sampelowany DataFrame
    """
    if len(df) <= n_rows:
        return df
    
    # Deterministyczny head (szybki, reproducible)
    return df.head(n_rows)


def build_data_profile(
    df: pd.DataFrame,
    max_cols: int = MAX_PROFILE_COLS,
    max_rows: int = MAX_PROFILE_ROWS
) -> DataProfile:
    """
    Buduje profil danych (bez PII).
    
    Args:
        df: DataFrame do profilowania
        max_cols: Maksymalna liczba kolumn
        max_rows: Maksymalna liczba wierszy
        
    Returns:
        DataProfile z metrykami
    """
    # Ogranicz kolumny i wiersze
    df_sample = df.iloc[:, :max_cols]
    df_sample = sample_for_profile(df_sample, max_rows)
    
    # Podstawowe metryki
    total_cells = df_sample.size or 1
    missing_count = df_sample.isna().sum().sum()
    missing_pct = float(missing_count / total_cells * 100.0)
    dupes = int(df_sample.duplicated().sum())
    
    # Typy kolumn
    type_counts = df_sample.dtypes.astype(str).value_counts().to_dict()
    
    # Top missing columns
    missing_by_col = df_sample.isna().mean()
    top_missing = missing_by_col[missing_by_col > 0].sort_values(ascending=False)
    top_missing_dict = {
        str(col): float(pct)
        for col, pct in top_missing.head(TOP_MISSING).items()
    }
    
    # Top nunique columns
    nunique_by_col = df_sample.nunique(dropna=True)
    top_nunique = nunique_by_col.sort_values(ascending=False)
    top_nunique_dict = {
        str(col): int(count)
        for col, count in top_nunique.head(TOP_NUNIQUE).items()
    }
    
    # Numeric variance
    numeric_cols = df_sample.select_dtypes(include="number")
    numeric_variance = {}
    
    if not numeric_cols.empty:
        variance_series = numeric_cols.var(numeric_only=True)
        top_var = variance_series.sort_values(ascending=False)
        numeric_variance = {
            str(col): float(var)
            for col, var in top_var.head(TOP_NUMERIC_VAR).items()
        }
    
    return DataProfile(
        rows=len(df_sample),
        cols=df_sample.shape[1],
        missing_pct=missing_pct,
        dupes=dupes,
        type_counts=type_counts,
        top_missing=top_missing_dict,
        top_nunique=top_nunique_dict,
        numeric_variance=numeric_variance
    )


def profile_to_text(profile: DataProfile, full_rows: int, full_cols: int) -> str:
    """
    Konwertuje profil do tekstu dla promptu.
    
    Args:
        profile: DataProfile
        full_rows: Pełna liczba wierszy (przed samplingiem)
        full_cols: Pełna liczba kolumn
        
    Returns:
        Tekstowa reprezentacja profilu
    """
    lines = []
    
    # Podstawowe info
    lines.append(f"- Wiersze: {profile.rows:,} (z {full_rows:,})")
    lines.append(f"- Kolumny: {profile.cols:,} (z {full_cols:,})")
    lines.append(f"- Braki: {profile.missing_pct:.2f}%")
    lines.append(f"- Duplikaty: {profile.dupes:,}")
    
    # Typy
    type_str = ", ".join([f"{t}: {n}" for t, n in profile.type_counts.items()])
    lines.append(f"- Typy kolumn: {type_str}")
    
    # Top missing
    if profile.top_missing:
        missing_str = ", ".join([
            f"{col} ({pct:.1%})"
            for col, pct in profile.top_missing.items()
        ])
        lines.append(f"- Najwięcej braków: {missing_str}")
    
    # Top nunique
    if profile.top_nunique:
        nunique_str = ", ".join([
            f"{col}: {count}"
            for col, count in profile.top_nunique.items()
        ])
        lines.append(f"- Top unikalnych wartości: {nunique_str}")
    
    # Variance
    if profile.numeric_variance:
        var_str = ", ".join([
            f"{col}: {var:.2e}"
            for col, var in profile.numeric_variance.items()
        ])
        lines.append(f"- Największa wariancja: {var_str}")
    
    return "\n".join(lines)


# ========================================================================================
# PROMPT BUILDING
# ========================================================================================

def build_user_prompt(
    profile_text: str,
    goal: Optional[str],
    mode: str,
    domain_hint: Optional[str]
) -> str:
    """
    Buduje prompt użytkownika.
    
    Args:
        profile_text: Tekstowy profil danych
        goal: Cel biznesowy
        mode: Tryb (standard/executive/technical)
        domain_hint: Podpowiedź domeny
        
    Returns:
        Prompt string
    """
    goal_text = goal.strip() if goal else "Brak sprecyzowanego celu."
    
    # Ton zależny od trybu
    tone_map = {
        "standard": "styl zbalansowany: klarowny i biznesowy",
        "executive": "styl executive: bardzo zwięzły, nacisk na wpływ/ROI i KPI",
        "technical": "styl techniczny: precyzyjny, metryki i metody"
    }
    
    tone = tone_map.get(mode, tone_map["standard"])
    
    # Domain line
    domain_line = f"Domena: {domain_hint}\n" if domain_hint else ""
    
    prompt = f"""Dane (profil):
{profile_text}

Cel użytkownika:
{goal_text}

{domain_line}Tryb pisania: {tone}

Zadanie:
1) Napisz krótkie executive summary (2-3 zdania)
2) Podaj 3-6 najważniejszych insightów związanych z celem
3) Zaproponuj 2-4 rekomendacje biznesowe / KPI do monitorowania
4) Wskaż potencjalne ryzyka / ograniczenia danych

WAŻNE:
- Zwróć WYŁĄCZNIE JSON zgodny ze schematem (bez komentarzy)
- Każdy wpis listy ≤ 180 znaków, bez markdown i emoji
- Nie wymyślaj liczb — odwołuj się do trendów wynikających z profilu
- Bądź konkretny i actionable

Format JSON:
{{
  "summary": "string",
  "top_insights": ["string", ...],
  "recommendations": ["string", ...],
  "risks": ["string", ...]
}}"""
    
    return prompt


# ========================================================================================
# API CALL WITH RETRY
# ========================================================================================

def call_api_with_retry(
    system: str,
    user: str,
    max_retries: int = MAX_RETRIES,
    timeout: int = API_TIMEOUT
) -> Tuple[Optional[str], int]:
    """
    Wywołuje API z retry logic.
    
    Args:
        system: System prompt
        user: User prompt
        max_retries: Maksymalna liczba prób
        timeout: Timeout w sekundach
        
    Returns:
        Tuple (response, retry_count)
    """
    last_error = None
    
    for attempt in range(max_retries):
        try:
            LOGGER.info(f"API call attempt {attempt + 1}/{max_retries}")
            
            response = chat_completion(
                system=system,
                user=user,
                response_format="json"
            )
            
            LOGGER.info(f"API call successful on attempt {attempt + 1}")
            return response, attempt
            
        except Exception as e:
            last_error = e
            LOGGER.warning(f"API call failed (attempt {attempt + 1}): {e}")
            
            if attempt < max_retries - 1:
                # Exponential backoff
                wait_time = RETRY_DELAY * (RETRY_BACKOFF ** attempt)
                LOGGER.info(f"Retrying in {wait_time}s...")
                time.sleep(wait_time)
    
    # Wszystkie próby zawiodły
    LOGGER.error(f"All {max_retries} API call attempts failed")
    raise Exception(f"API call failed after {max_retries} attempts: {last_error}")


# ========================================================================================
# GŁÓWNA FUNKCJA
# ========================================================================================

def generate_insights(
    df: pd.DataFrame,
    goal: Optional[str] = None,
    mode: Literal["standard", "executive", "technical"] = "standard",
    *,
    options: Optional[InsightsOptions] = None,
    use_cache: bool = True
) -> Dict[str, Any]:
    """
    Generuje insighty AI w stabilnym formacie JSON.
    
    Args:
        df: DataFrame z danymi
        goal: Cel biznesowy (opcjonalny)
        mode: Tryb generowania (standard/executive/technical)
        options: Dodatkowe opcje (InsightsOptions)
        use_cache: Czy użyć cache
        
    Returns:
        Słownik zgodny ze schematem:
        {
            "summary": str,
            "top_insights": List[str],
            "recommendations": List[str],
            "risks": List[str],
            "cached": bool (opcjonalne),
            "retries": int (opcjonalne)
        }
    """
    # Opcje
    opts = options or InsightsOptions(mode=mode)
    
    # Walidacja danych
    if not isinstance(df, pd.DataFrame):
        LOGGER.error("Input is not a DataFrame")
        return _fallback_response("Input nie jest DataFrame")
    
    if df.empty:
        LOGGER.warning("DataFrame is empty")
        return _fallback_response("Brak danych do analizy")
    
    # Cache check
    if use_cache:
        df_hash = compute_dataframe_hash(df)
        cache_key = compute_cache_key(df_hash, goal, mode, opts)
        
        cached_result = _INSIGHTS_CACHE.get(cache_key)
        if cached_result is not None:
            LOGGER.info("Returning cached insights")
            result = cached_result.copy()
            result["cached"] = True
            return result
    
    # Build profile
    try:
        LOGGER.info("Building data profile")
        profile = build_data_profile(
            df=df,
            max_cols=opts.max_profile_cols,
            max_rows=opts.sample_rows_for_profile
        )
        
        profile_text = profile_to_text(profile, len(df), df.shape[1])
        
    except Exception as e:
        LOGGER.error(f"Failed to build profile: {e}", exc_info=True)
        return _fallback_response("Błąd budowania profilu danych")
    
    # Build prompt
    user_prompt = build_user_prompt(profile_text, goal, opts.mode, opts.domain_hint)
    
    # Call API with retry
    try:
        LOGGER.info("Calling OpenAI API")
        raw_response, retry_count = call_api_with_retry(
            system=SYSTEM_PROMPT,
            user=user_prompt,
            max_retries=MAX_RETRIES,
            timeout=opts.timeout
        )
        
    except Exception as e:
        LOGGER.error(f"API call failed completely: {e}", exc_info=True)
        return _fallback_response("Błąd komunikacji z API")
    
    # Parse response
    try:
        payload = _parse_and_validate_response(raw_response, user_prompt)
        
        if payload is None:
            LOGGER.error("Failed to parse valid response")
            return _fallback_response(
                "Nie udało się uzyskać poprawnego JSON",
                raw_text=raw_response
            )
        
    except Exception as e:
        LOGGER.error(f"Response parsing failed: {e}", exc_info=True)
        return _fallback_response("Błąd parsowania odpowiedzi")
    
    # Clean and truncate
    summary = truncate_string(payload.summary.strip(), MAX_SUMMARY_LENGTH)
    top_insights = cap_list_items(payload.top_insights, MAX_ITEM_LENGTH, MAX_LIST_ITEMS)
    recommendations = cap_list_items(payload.recommendations, MAX_ITEM_LENGTH, MAX_LIST_ITEMS)
    risks = cap_list_items(payload.risks, MAX_ITEM_LENGTH, MAX_LIST_ITEMS)
    
    # Build result
    result = {
        "summary": summary,
        "top_insights": top_insights,
        "recommendations": recommendations,
        "risks": risks,
        "cached": False,
        "retries": retry_count
    }
    
    # Cache result
    if use_cache:
        _INSIGHTS_CACHE.set(cache_key, result.copy())
    
    LOGGER.info("Insights generated successfully")
    return result


def _parse_and_validate_response(
    raw_response: str,
    user_prompt: str
) -> Optional[InsightsPayload]:
    """
    Parsuje i waliduje odpowiedź API z retry.
    
    Args:
        raw_response: Surowa odpowiedź
        user_prompt: Prompt użytkownika (dla retry)
        
    Returns:
        InsightsPayload lub None
    """
    # First attempt: direct JSON parse
    parsed = safe_json_loads(raw_response)
    
    if parsed is None:
        # Try to extract JSON from text
        LOGGER.warning("Direct JSON parse failed, trying extraction")
        json_text = extract_json_from_text(raw_response)
        
        if json_text:
            parsed = safe_json_loads(json_text)
    
    # Validate schema
    if parsed is not None:
        payload = coerce_to_schema(parsed)
        
        if payload is not None:
            return payload
    
    # Retry z dodatkową instrukcją
    LOGGER.warning("Invalid response format, attempting retry with hint")
    
    retry_prompt = user_prompt + "\n\nUWAGA: Poprzednia odpowiedź naruszała format. Zwróć 100% poprawny JSON — nic poza JSON."
    
    try:
        raw_retry, _ = call_api_with_retry(
            system=SYSTEM_PROMPT,
            user=retry_prompt,
            max_retries=1  # Only 1 retry for correction
        )
        
        # Parse retry
        parsed_retry = safe_json_loads(raw_retry)
        
        if parsed_retry is None:
            json_text = extract_json_from_text(raw_retry)
            if json_text:
                parsed_retry = safe_json_loads(json_text)
        
        if parsed_retry is not None:
            return coerce_to_schema(parsed_retry)
            
    except Exception as e:
        LOGGER.error(f"Retry failed: {e}")
    
    return None


def _fallback_response(message: str, raw_text: Optional[str] = None) -> Dict[str, Any]:
    """
    Tworzy fallback response przy błędzie.
    
    Args:
        message: Komunikat błędu
        raw_text: Surowy tekst (opcjonalny)
        
    Returns:
        Minimalny poprawny response
    """
    result = {
        "summary": message,
        "top_insights": [],
        "recommendations": ["Sprawdź dane wejściowe", "Spróbuj ponownie"],
        "risks": ["Brak danych do analizy"],
        "cached": False,
        "retries": 0
    }
    
    if raw_text:
        result["raw_text"] = raw_text
    
    return result


# ========================================================================================
# CACHE MANAGEMENT
# ========================================================================================

def clear_insights_cache() -> None:
    """Czyści cache insightów."""
    _INSIGHTS_CACHE.clear()
    LOGGER.info("Insights cache cleared")


def get_cache_stats() -> Dict[str, Any]:
    """
    Zwraca statystyki cache.
    
    Returns:
        Słownik ze statystykami
    """
    return {
        "size": len(_INSIGHTS_CACHE._cache),
        "ttl": _INSIGHTS_CACHE.ttl
    }