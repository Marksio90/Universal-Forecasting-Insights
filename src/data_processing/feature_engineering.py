from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Iterable
import logging
import pandas as pd
import numpy as np

__all__ = ["DateFeatureConfig", "detect_datetime_cols", "basic_features"]

log = logging.getLogger(__name__)

@dataclass
class DateFeatureConfig:
    date_cols: Optional[List[str]] = None      # jeśli None → auto-detekcja
    parse_dates: bool = False                  # parsuj object/string → datetime
    dayfirst: bool = False                     # PL: True, jeśli "dd-mm-rrrr"
    tz_convert_to: Optional[str] = None        # np. "Europe/Warsaw"
    add_calendar: bool = True                  # year, quarter, month, day, dow, doy
    add_flags: bool = True                     # is_month_start/end, is_quarter_start/end, is_year_start/end, is_weekend
    add_clock: bool = True                     # hour, minute, second (jeśli obecny czas)
    add_cyclic: bool = True                    # sin/cos dla month/dow/hour
    add_holidays: bool = False                 # wymaga pakietu "holidays"
    holidays_country: str = "PL"
    max_new_cols: int = 256                    # bezpieczny limit rozszerzeń
    prefix_sep: str = "_"                      # separator w nazwach

def detect_datetime_cols(df: pd.DataFrame, *, limit: Optional[int] = None) -> List[str]:
    """Wykryj kolumny datowe (dtype datetime lub heurystyka stringów z wieloma sukcesami parsowania)."""
    cols: List[str] = []
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_datetime64_any_dtype(s):
            cols.append(c); continue
        # heurystyka tylko dla object/string i krótkich kolumn
        if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
            sample = s.dropna().astype(str).head(500)
            if sample.empty: 
                continue
            parsed = pd.to_datetime(sample, errors="coerce", dayfirst=False, utc=False)
            if parsed.notna().mean() >= 0.8:
                cols.append(c)
        if limit and len(cols) >= limit:
            break
    return cols

def _ensure_datetime(s: pd.Series, cfg: DateFeatureConfig) -> pd.Series:
    """Zwraca kolumnę jako datetime64 (naive lub skonwertowaną do strefy), bezpiecznie."""
    x = s
    if not pd.api.types.is_datetime64_any_dtype(x) and cfg.parse_dates:
        x = pd.to_datetime(x, errors="coerce", dayfirst=cfg.dayfirst, utc=False)
    # konwersja strefy: obsłuż tz-aware i tz-naive
    if cfg.tz_convert_to:
        try:
            if getattr(x.dt, "tz", None) is None:
                x = x.dt.tz_localize(cfg.tz_convert_to, nonexistent="NaT", ambiguous="NaT")
            else:
                x = x.dt.tz_convert(cfg.tz_convert_to)
            x = x.dt.tz_localize(None)  # zrzucamy tz do naive, by uniknąć propagacji TZ w downstream
        except Exception:
            pass
    return x

def _safe_name(base: str, suffix: str, sep: str) -> str:
    return f"{base}{sep}{suffix}"

def _add(df: pd.DataFrame, name: str, values: Any, out_cols: List[str]) -> None:
    if name in df.columns:
        # unikaj kolizji — dołóż licznik
        i = 1
        nn = f"{name}__{i}"
        while nn in df.columns:
            i += 1; nn = f"{name}__{i}"
        name = nn
    df[name] = values
    out_cols.append(name)

def _cyc_encode(series: pd.Series, period: int) -> Dict[str, pd.Series]:
    # normalizacja 1..period lub 0..period-1
    vals = series.astype(float)
    # jeśli wartości zaczynają się od 0 (np. dow), przesunięcie o 0.5 poprawia gładkość
    sin = np.sin(2 * np.pi * vals / period)
    cos = np.cos(2 * np.pi * vals / period)
    return {"sin": sin, "cos": cos}

def basic_features(df: pd.DataFrame, config: Optional[DateFeatureConfig] = None) -> pd.DataFrame:
    """
    Rozszerza DataFrame o cechy czasowe. Wstecznie kompatybilne z prostą wersją (year/month/dow),
    a dodatkowo: quarter, day, dayofyear, flagi start/end, weekend, hour/minute/second oraz
    cykliczne kodowania sin/cos i (opcjonalnie) święta państwowe.

    Parametry sterujesz przez DateFeatureConfig. Bez configu działa sensownie out-of-the-box.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("basic_features: expected pandas.DataFrame")
    cfg = config or DateFeatureConfig()
    out = df.copy()
    new_cols: List[str] = []

    # 1) wybór kolumn datowych
    date_cols = (cfg.date_cols or detect_datetime_cols(out))[:cfg.max_new_cols]
    if not date_cols:
        return out

    # ewentualna inicjalizacja libki świąt
    _hol = None
    if cfg.add_holidays and len(out) > 0:
        try:
            import holidays  # type: ignore
            # wybierz lata występujące w danych, by nie rozdmuchiwać pamięci
            # (wyznaczymy po pierwszej kolumnie, reszta ma zbliżony zakres)
            s0 = _ensure_datetime(out[date_cols[0]], cfg)
            years = sorted(set(s0.dropna().dt.year.tolist()))
            _hol = holidays.country_holidays(cfg.holidays_country, years=years) if years else None
        except Exception:
            _hol = None
            log.debug("Holidays not available; skipping.")

    # 2) generacja cech
    for c in date_cols:
        sdt = _ensure_datetime(out[c], cfg)
        if not pd.api.types.is_datetime64_any_dtype(sdt):
            # nie udało się ustandaryzować — pomiń kolumnę
            continue

        base = c
        # kalendarz
        if cfg.add_calendar:
            _add(out, _safe_name(base, "year", cfg.prefix_sep), sdt.dt.year.astype("Int64"), new_cols)
            _add(out, _safe_name(base, "quarter", cfg.prefix_sep), sdt.dt.quarter.astype("Int64"), new_cols)
            _add(out, _safe_name(base, "month", cfg.prefix_sep), sdt.dt.month.astype("Int64"), new_cols)
            _add(out, _safe_name(base, "day", cfg.prefix_sep), sdt.dt.day.astype("Int64"), new_cols)
            _add(out, _safe_name(base, "dow", cfg.prefix_sep), sdt.dt.dayofweek.astype("Int64"), new_cols)  # 0=Mon
            _add(out, _safe_name(base, "doy", cfg.prefix_sep), sdt.dt.dayofyear.astype("Int64"), new_cols)

        # flagi start/end + weekend
        if cfg.add_flags:
            _add(out, _safe_name(base, "is_month_start", cfg.prefix_sep), sdt.dt.is_month_start.astype("boolean"), new_cols)
            _add(out, _safe_name(base, "is_month_end", cfg.prefix_sep), sdt.dt.is_month_end.astype("boolean"), new_cols)
            _add(out, _safe_name(base, "is_quarter_start", cfg.prefix_sep), sdt.dt.is_quarter_start.astype("boolean"), new_cols)
            _add(out, _safe_name(base, "is_quarter_end", cfg.prefix_sep), sdt.dt.is_quarter_end.astype("boolean"), new_cols)
            _add(out, _safe_name(base, "is_year_start", cfg.prefix_sep), sdt.dt.is_year_start.astype("boolean"), new_cols)
            _add(out, _safe_name(base, "is_year_end", cfg.prefix_sep), sdt.dt.is_year_end.astype("boolean"), new_cols)
            # weekend: sob(5), niedz(6)
            weekend = sdt.dt.dayofweek.isin([5, 6]).astype("boolean")
            _add(out, _safe_name(base, "is_weekend", cfg.prefix_sep), weekend, new_cols)

        # zegar (tylko jeśli dane mają komponent czasu różny od 00:00)
        if cfg.add_clock:
            has_time = (sdt.dt.hour != 0).any() or (sdt.dt.minute != 0).any() or (sdt.dt.second != 0).any()
            if has_time:
                _add(out, _safe_name(base, "hour", cfg.prefix_sep), sdt.dt.hour.astype("Int64"), new_cols)
                _add(out, _safe_name(base, "minute", cfg.prefix_sep), sdt.dt.minute.astype("Int64"), new_cols)
                _add(out, _safe_name(base, "second", cfg.prefix_sep), sdt.dt.second.astype("Int64"), new_cols)

        # kodowanie cykliczne
        if cfg.add_cyclic:
            # month ∈ [1..12]
            m_col = _safe_name(base, "month", cfg.prefix_sep)
            if m_col in out.columns:
                cyc = _cyc_encode(out[m_col].astype(float), 12)
                _add(out, _safe_name(base, "month_sin", cfg.prefix_sep), cyc["sin"], new_cols)
                _add(out, _safe_name(base, "month_cos", cfg.prefix_sep), cyc["cos"], new_cols)

            # dow ∈ [0..6] → przesunięcie +1 dla pełnej skali 1..7 (nie jest konieczne, ale spójne)
            d_col = _safe_name(base, "dow", cfg.prefix_sep)
            if d_col in out.columns:
                cyc = _cyc_encode(out[d_col].astype(float), 7)
                _add(out, _safe_name(base, "dow_sin", cfg.prefix_sep), cyc["sin"], new_cols)
                _add(out, _safe_name(base, "dow_cos", cfg.prefix_sep), cyc["cos"], new_cols)

            # hour ∈ [0..23]
            h_col = _safe_name(base, "hour", cfg.prefix_sep)
            if h_col in out.columns:
                cyc = _cyc_encode(out[h_col].astype(float), 24)
                _add(out, _safe_name(base, "hour_sin", cfg.prefix_sep), cyc["sin"], new_cols)
                _add(out, _safe_name(base, "hour_cos", cfg.prefix_sep), cyc["cos"], new_cols)

        # święta (jeśli biblioteka dostępna i udało się zainicjalizować)
        if _hol is not None:
            try:
                is_hol = sdt.dt.date.map(lambda d: d in _hol if pd.notna(d) else False).astype("boolean")
                _add(out, _safe_name(base, "is_holiday", cfg.prefix_sep), is_hol, new_cols)
            except Exception:
                pass

        # limit bezpieczeństwa
        if len(new_cols) >= cfg.max_new_cols:
            log.warning("basic_features: reached max_new_cols=%d; stopping further expansions.", cfg.max_new_cols)
            break

    return out
