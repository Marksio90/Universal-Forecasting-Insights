# src/utils/cache.py
from __future__ import annotations
from functools import lru_cache, wraps
from typing import Callable, Optional, TypeVar, Any
import os

T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])

# === Konfiguracja z ENV (global switches) ===
_CACHE_ENABLED = (os.getenv("DATAGENIUS_CACHE_ENABLED", "1").strip().lower() in {"1", "true", "yes"})
_DEFAULT_TTL_MIN = int(os.getenv("DATAGENIUS_CACHE_TTL_MIN", "0"))  # 0 = bez TTL

# Streamlit jest opcjonalny – moduł działa również poza Streamlitem
try:
    import streamlit as st  # type: ignore
    _HAS_ST = True
except Exception:
    st = None  # type: ignore
    _HAS_ST = False


# === Dekorator cache_data ===
def cache_data(
    _func: Optional[F] = None,
    *,
    ttl: Optional[int] = None,
    show_spinner: bool = False,
    persist: bool = False,
    max_entries: Optional[int] = None,
    fallback_lru: Optional[int] = 256,
) -> Callable[[F], F]:
    """
    Cache wyników funkcji operujących na danych (wyniki zależne od argumentów).
    - W Streamlit używa st.cache_data(ttl/persist/show_spinner/max_entries).
    - Gdy Streamlit niedostępny lub cache wyłączony ENV-em → fallback do lru_cache (lub no-op, jeśli fallback_lru=None).

    Użycie:
      @cache_data                           # bez parametrów
      @cache_data(ttl=600, persist=True)    # z parametrami

    Parametry:
      ttl:        sekundowy TTL cache'u (jeśli None, używa DATAGENIUS_CACHE_TTL_MIN*60 gdy >0)
      show_spinner: wyświetla spinner w Streamlit
      persist:    utrwala cache między restartami (wymaga ustawień Streamlit)
      max_entries: limit liczby różnych kluczy w cache Streamlit
      fallback_lru: rozmiar fallback LRU poza Streamlitem; None = wyłącz fallback (no-op)
    """
    def _decorate(func: F) -> F:
        nonlocal ttl
        # Domyślny TTL z ENV (minuty -> sekundy)
        if ttl is None and _DEFAULT_TTL_MIN > 0:
            ttl = _DEFAULT_TTL_MIN * 60

        if _HAS_ST and _CACHE_ENABLED:
            # Streamlit – prawdziwy cache danych
            wrapped = st.cache_data(
                ttl=ttl,
                show_spinner=show_spinner,
                persist=persist,
                max_entries=max_entries,
            )(func)
            return wrapped  # type: ignore[return-value]

        # Fallback: poza Streamlit lub cache OFF
        if fallback_lru is None:
            # no-op: brak cache
            return func
        return lru_cache(maxsize=fallback_lru)(func)  # type: ignore[return-value]

    # wspiera @cache_data oraz @cache_data(...)
    return _decorate if _func is None else _decorate(_func)


# === Dekorator cache_resource ===
def cache_resource(
    _func: Optional[F] = None,
    *,
    show_spinner: bool = False,
    fallback_singleton: bool = True,
) -> Callable[[F], F]:
    """
    Cache zasobów (modele, połączenia) – pojedyncza instancja na proces/sesję.
    - W Streamlit: st.cache_resource(show_spinner=...).
    - Poza Streamlit: prosty singleton (zapamiętanie zwróconej wartości).

    Użycie:
      @cache_resource
      def load_model(): ...

    Parametry:
      show_spinner: spinner w Streamlit
      fallback_singleton: gdy True, poza Streamlit zwracaj zawsze tę samą instancję (pierwszy wynik).
    """
    def _decorate(func: F) -> F:
        if _HAS_ST and _CACHE_ENABLED:
            wrapped = st.cache_resource(show_spinner=show_spinner)(func)
            return wrapped  # type: ignore[return-value]

        if not fallback_singleton:
            return func

        # Prosty singleton poza Streamlit
        _SENTINEL = object()
        _cache_val: Any = _SENTINEL

        @wraps(func)
        def _wrapper(*args: Any, **kwargs: Any) -> Any:
            nonlocal _cache_val
            if _cache_val is _SENTINEL:
                _cache_val = func(*args, **kwargs)
            return _cache_val

        return _wrapper  # type: ignore[return-value]

    return _decorate if _func is None else _decorate(_func)


# === Proste memoize (LRU) dla stringowych kluczy ===
@lru_cache(maxsize=128)
def memoize(key: str) -> str:
    """
    Lekki, deterministyczny memoizer „klucz->klucz”.
    Przydatny do buforowania drobnych transformacji.
    """
    return key


# === Czyszczenie cache ===
def clear_data_cache() -> None:
    """Czyści cache danych (Streamlit). Poza Streamlitem – brak efektu."""
    if _HAS_ST:
        try:
            st.cache_data.clear()
        except Exception:
            pass

def clear_resource_cache() -> None:
    """Czyści cache zasobów (Streamlit). Poza Streamlitem – brak efektu."""
    if _HAS_ST:
        try:
            st.cache_resource.clear()
        except Exception:
            pass

def clear_all_caches() -> None:
    """Czyści oba cache w Streamlit (jeśli dostępne)."""
    clear_data_cache()
    clear_resource_cache()


# === Szybkie helpery statusu ===
def cache_enabled() -> bool:
    """Czy globalny cache jest włączony (ENV: DATAGENIUS_CACHE_ENABLED)."""
    return _CACHE_ENABLED

def has_streamlit() -> bool:
    """Czy dostępny jest Streamlit (wpływa na backend cache)."""
    return _HAS_ST
