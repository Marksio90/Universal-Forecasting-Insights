# src/utils/helpers.py
# === KONTEKST ===
# Uniwersalny, defensywny loader danych dla Streamlit/FastAPI/CLI.
# Obsługuje: CSV/TSV (auto-sep), Parquet/Feather/ORC, Excel, JSON/JSONL, pliki skompresowane.
# Robi: detekcję formatu/enkodowania, parsing dat (auto dayfirst), downcast typów, deduplikację kolumn, normalizację NaN.
# Zwraca: DataFrame oraz opcjonalnie metadane (LoadMeta).

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, Any, Dict, List
import io, os, time, logging, hashlib
import pandas as pd
import numpy as np

# PyArrow dla kolumnarów (Parquet/Feather/ORC) – jeśli nie ma, pandas spróbuje fallback
try:
    import pyarrow  # noqa: F401
except Exception:
    pyarrow = None  # type: ignore

# Opcjonalna detekcja enkodowania
try:
    import chardet
except Exception:
    chardet = None  # type: ignore

logger = logging.getLogger("datagenius.helpers")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# === META ===
@dataclass
class LoadMeta:
    source_name: Optional[str]
    detected_format: str
    encoding: Optional[str]
    rows: int
    cols: int
    parsed_dates: List[str]
    dtypes_before: Dict[str, str]
    dtypes_after: Dict[str, str]
    read_time_ms: int

# === NARZĘDZIA WEWNĘTRZNE ===
def _get_name(obj: Any) -> Optional[str]:
    return getattr(obj, "name", None) or getattr(obj, "filename", None) or getattr(obj, "path", None)

def _read_all_bytes(obj: Any) -> bytes:
    # Obsługa: UploadedFile (Streamlit), file-like, ścieżka, bytes
    if isinstance(obj, (bytes, bytearray)):
        return bytes(obj)
    if isinstance(obj, str) and os.path.exists(obj):
        with open(obj, "rb") as f:
            return f.read()
    if hasattr(obj, "read"):
        # Streamy mogą być na końcu – spróbuj cofnąć
        try:
            obj.seek(0)
        except Exception:
            pass
        data = obj.read()
        # Przywróć kursor (grzecznościowo)
        try:
            obj.seek(0)
        except Exception:
            pass
        return data if isinstance(data, (bytes, bytearray)) else bytes(data)
    raise TypeError("Unsupported input type for smart_read/load_dataframe")

def _detect_format(name: Optional[str], head: bytes) -> str:
    nm = (name or "").lower()
    # po rozszerzeniu
    for ext, fmt in {
        ".parquet": "parquet",
        ".feather": "feather",
        ".feather64": "feather",
        ".orc": "orc",
        ".xlsx": "excel",
        ".xls": "excel",
        ".xlsm": "excel",
        ".jsonl": "jsonl",
        ".ndjson": "jsonl",
        ".json": "json",
        ".csv": "csv",
        ".tsv": "tsv",
        ".txt": "csv",
        ".gz": "maybe_compressed",
        ".zip": "maybe_compressed",
        ".bz2": "maybe_compressed",
        ".xz": "maybe_compressed",
    }.items():
        if nm.endswith(ext):
            if fmt == "maybe_compressed":
                # spróbuj wykryć wewnętrzny format po nagłówku – tu i tak pandas ogarnie
                return "csv"  # domyśl po tekście; kolumnarne i excel zwykle nie w .gz
            return fmt
    # po nagłówku
    sig = head[:8]
    if sig.startswith(b"PK"):  # zip/xlsx/feather
        return "excel" if nm.endswith((".xlsx", ".xlsm")) else "csv"
    if sig == b"PAR1\x15\x04\x15\x04":
        return "parquet"
    # fallback
    return "csv"

def _detect_encoding(raw: bytes) -> Optional[str]:
    if chardet is None:
        # spróbuj UTF-8 z podpisem; jeśli błąd, użyj latin-1 (bezpieczny superset)
        try:
            raw.decode("utf-8")
            return "utf-8"
        except Exception:
            return "latin-1"
    res = chardet.detect(raw[:200_000])
    enc = res.get("encoding")
    if enc:
        return enc
    # fallback
    return "utf-8"

def _dedupe_columns(cols: List[str]) -> List[str]:
    seen: Dict[str, int] = {}
    out: List[str] = []
    for c in cols:
        if c not in seen:
            seen[c] = 0
            out.append(c)
        else:
            seen[c] += 1
            out.append(f"{c}__{seen[c]}")
    return out

def _downcast_numbers(df: pd.DataFrame) -> pd.DataFrame:
    # bezpieczny downcast int/float -> zmniejszenie pamięci
    for c in df.select_dtypes(include=["int64", "int32", "int16"]).columns:
        df[c] = pd.to_numeric(df[c], downcast="integer")
    for c in df.select_dtypes(include=["float64", "float32"]).columns:
        df[c] = pd.to_numeric(df[c], downcast="float")
    return df

def _normalize_nans(df: pd.DataFrame) -> pd.DataFrame:
    # ujednolicenie typowych wskaźników braków (po CSV/Excel)
    na_like = {"", "na", "n/a", "none", "null", "nan", "-", "--"}
    obj_cols = df.select_dtypes(include=["object"]).columns
    for c in obj_cols:
        df[c] = df[c].replace({v: np.nan for v in na_like})
    return df

def _auto_parse_dates(df: pd.DataFrame, threshold: float = 0.8) -> Tuple[pd.DataFrame, List[str]]:
    parsed_cols: List[str] = []
    obj_cols = df.select_dtypes(include=["object"]).columns
    for c in obj_cols:
        s = df[c]
        # szybki test: jeśli większość wpisów zawiera cyfry lub '-'/'/'/':', spróbuj daty
        sample = s.dropna().astype(str).head(200).str.contains(r"[\d:/\-TZ ]", regex=True).mean()
        if sample < 0.5:
            continue
        # spróbuj dayfirst auto: wybierz lepszy wynik
        for dayfirst in (False, True):
            try:
                parsed = pd.to_datetime(s, errors="coerce", utc=False, dayfirst=dayfirst, infer_datetime_format=True)
                if parsed.notna().mean() >= threshold:
                    df[c] = parsed
                    parsed_cols.append(c)
                    break
            except Exception:
                continue
    return df, parsed_cols

def _features_after_preprocessing(df: pd.DataFrame) -> Dict[str, str]:
    return {c: str(t) for c, t in df.dtypes.items()}

def _hash_for_cache(raw: bytes, name: Optional[str]) -> str:
    h = hashlib.sha256()
    h.update(raw[:10_000_000])  # hashuj do 10MB na potrzeby cache (wystarczy do rozróżnienia)
    if name:
        h.update(name.encode("utf-8"))
    return h.hexdigest()

# === PUBLICZNE API ===
def smart_read(obj: Any, *, return_meta: bool = False) -> pd.DataFrame | Tuple[pd.DataFrame, LoadMeta]:
    """
    Przeczytaj plik dowolnego wspieranego formatu do DataFrame.
    - obj: UploadedFile/file-like/bytes/ścieżka
    - return_meta: jeśli True, zwróci (df, LoadMeta)
    """
    t0 = time.time()
    raw = _read_all_bytes(obj)
    name = _get_name(obj)
    fmt = _detect_format(name, raw[:4096])
    enc = None
    dtypes_before: Dict[str, str] = {}
    parsed_dates: List[str] = []

    bio = io.BytesIO(raw)

    try:
        if fmt in {"parquet", "feather", "orc"}:
            if fmt == "parquet":
                df = pd.read_parquet(bio)
            elif fmt == "feather":
                df = pd.read_feather(bio)
            else:
                # pandas nie ma read_orc – użyj pyarrow.orc -> to_pandas
                import pyarrow.orc as pa_orc  # type: ignore
                df = pa_orc.ORCFile(bio).read().to_pandas()
        elif fmt == "excel":
            df = pd.read_excel(bio, sheet_name=0, engine=None)  # auto-engine
        elif fmt in {"json", "jsonl"}:
            text = raw.decode("utf-8", errors="replace")
            if fmt == "jsonl":
                df = pd.read_json(io.StringIO(text), lines=True)
            else:
                # Spróbuj obiektu list/dict
                try:
                    df = pd.read_json(io.StringIO(text), orient="records")
                except ValueError:
                    df = pd.json_normalize(pd.read_json(io.StringIO(text)))
        else:
            # CSV/TSV/anything text-like
            enc = _detect_encoding(raw)
            text_b = raw.decode(enc, errors="replace")
            # sep=None -> python engine sniffuje separator; na dużych plikach można przekazać sep eksplicytnie
            df = pd.read_csv(io.StringIO(text_b), sep=None, engine="python")
    except Exception as e:
        raise RuntimeError(f"smart_read failed ({fmt}): {e}")

    # Dedupe headery
    df.columns = _dedupe_columns([str(c) for c in df.columns])
    dtypes_before = {c: str(t) for c, t in df.dtypes.items()}

    # Normalizacja + lekkie odchudzenie
    df = _normalize_nans(df)
    df = _downcast_numbers(df)

    # Auto parse dates
    df, parsed_dates = _auto_parse_dates(df, threshold=0.8)

    meta = LoadMeta(
        source_name=name,
        detected_format=fmt,
        encoding=enc,
        rows=int(df.shape[0]),
        cols=int(df.shape[1]),
        parsed_dates=parsed_dates,
        dtypes_before=dtypes_before,
        dtypes_after=_features_after_preprocessing(df),
        read_time_ms=int((time.time() - t0) * 1000),
    )

    if return_meta:
        return df, meta
    return df


def load_dataframe(uploaded_file: Any, *, return_meta: bool = False) -> Optional[pd.DataFrame] | Tuple[pd.DataFrame, LoadMeta]:
    """
    Zgodne z Twoim dotychczasowym API:
      - Jeśli `uploaded_file` jest None → None
      - W innym wypadku: używa smart_read + auto-parse dat (progiem 0.8), jak wcześniej.
      - Użyj `return_meta=True`, aby dostać (df, LoadMeta).
    """
    if uploaded_file is None:
        return None

    df, meta = smart_read(uploaded_file, return_meta=True)
    logger.info(f"Loaded {meta.detected_format} rows={meta.rows} cols={meta.cols} parsed_dates={meta.parsed_dates}")

    return (df, meta) if return_meta else df
