# universal_parser.py — TURBO PRO
from __future__ import annotations
import io
import os
import csv
import json
import gzip
from dataclasses import dataclass
from typing import Tuple, Optional, Any, Dict, List

import pandas as pd

# Opcjonalne zależności (ciche)
try:
    import chardet  # type: ignore
except Exception:
    chardet = None  # type: ignore

try:
    from PyPDF2 import PdfReader  # type: ignore
except Exception:
    PdfReader = None  # type: ignore

try:
    from docx import Document  # type: ignore
except Exception:
    Document = None  # type: ignore


# =========================
# Obsługiwane typy
# =========================
# Zachowujemy stare i rozszerzamy bez psucia kompatybilności
SUPPORTED = {
    "csv", "tsv", "txt", "xlsx", "xls", "json", "docx", "pdf",
    "parquet", "feather", "ndjson", "csv.gz", "tsv.gz"
}

# =========================
# Opcje i wynik PRO (opcjonalne do użycia)
# =========================
@dataclass(frozen=True)
class ParseOptions:
    prefer_pyarrow: bool = True       # spróbuj dtype_backend="pyarrow" gdzie się da
    max_rows: Optional[int] = None    # twardy limit wierszy (None = brak)
    max_cols: Optional[int] = None    # twardy limit kolumn (None = brak)
    excel_sheet: Optional[str] = None # jeśli znasz nazwę arkusza — wymuś
    ndjson: bool = True               # wymuś lines=True dla *.ndjson
    random_state: int = 42            # na przyszłość (sampling)
    # CSV heurystyki
    csv_delimiters: Tuple[str, ...] = (",", ";", "\t", "|")
    csv_try_encodings: Tuple[str, ...] = ("utf-8", "utf-8-sig", "cp1250", "iso-8859-2", "latin-1")
    csv_sniff_bytes: int = 10_000
    # Bezpieczeństwo
    sanitize_inf: bool = True
    drop_empty_rows: bool = False

@dataclass(frozen=True)
class ParseResult:
    df: Optional[pd.DataFrame]
    text: Optional[str]
    warnings: List[str]
    meta: Dict[str, Any]


# =========================
# Helpers
# =========================
def _maybe_limit(df: pd.DataFrame, opts: ParseOptions) -> pd.DataFrame:
    if df is None or not isinstance(df, pd.DataFrame):
        return df
    if opts.max_cols is not None and df.shape[1] > opts.max_cols:
        df = df.iloc[:, : opts.max_cols]
    if opts.max_rows is not None and len(df) > opts.max_rows:
        df = df.head(opts.max_rows)
    return df

def _apply_common_fixes(df: pd.DataFrame, opts: ParseOptions) -> pd.DataFrame:
    if df is None or not isinstance(df, pd.DataFrame):
        return df
    if opts.drop_empty_rows:
        df = df.dropna(how="all")
    if opts.sanitize_inf:
        df = df.replace([float("inf"), float("-inf")], pd.NA)
    return df

def _dtype_backend_kwargs(opts: ParseOptions) -> Dict[str, Any]:
    if not opts.prefer_pyarrow:
        return {}
    # pandas >=2.0 wspiera dtype_backend="pyarrow"
    try:
        return {"dtype_backend": "pyarrow"}  # type: ignore
    except Exception:
        return {}

def _detect_encoding(buf: bytes, candidates: Tuple[str, ...], sniff_bytes: int) -> str:
    head = buf[:sniff_bytes]
    if chardet is not None:
        try:
            res = chardet.detect(head) or {}
            enc = (res.get("encoding") or "").lower()
            if enc:
                return enc
        except Exception:
            pass
    # fallback — spróbuj sekwencyjnie
    for enc in candidates:
        try:
            head.decode(enc)
            return enc
        except Exception:
            continue
    return "utf-8"

def _sniff_delimiter(sample_text: str, delimiters: Tuple[str, ...]) -> str:
    # dużo tabów ⇒ TSV
    if sample_text.count("\t") > max(sample_text.count(","), sample_text.count(";")) and sample_text.count("\t") >= 2:
        return "\t"
    try:
        dialect = csv.Sniffer().sniff(sample_text, delimiters=delimiters)
        return dialect.delimiter  # type: ignore
    except Exception:
        # heurystyka średnik vs przecinek
        return ";" if sample_text.count(";") > sample_text.count(",") else ","

def _read_csv_like(buf: bytes, sep_hint: Optional[str], opts: ParseOptions) -> pd.DataFrame:
    enc = _detect_encoding(buf, opts.csv_try_encodings, opts.csv_sniff_bytes)
    sample = buf[:opts.csv_sniff_bytes].decode(enc, errors="ignore")
    sep = sep_hint or _sniff_delimiter(sample, opts.csv_delimiters)

    # Spróbuj z rozsądnymi defaultami
    for quoting in (csv.QUOTE_MINIMAL, csv.QUOTE_NONE):
        try:
            df = pd.read_csv(
                io.BytesIO(buf),
                sep=sep,
                encoding=enc,
                low_memory=False,
                quoting=quoting,
                **_dtype_backend_kwargs(opts),
            )
            return df
        except Exception:
            continue

    # Ostatnia próba — pozwól pandasowi zgadnąć
    return pd.read_csv(io.BytesIO(buf), low_memory=False, encoding=enc, **_dtype_backend_kwargs(opts))

def _read_xlsx(file_obj: io.BytesIO, opts: ParseOptions) -> pd.DataFrame:
    # próbujemy wszystkie arkusze i wybieramy największy niepusty, chyba że podano konkretny
    try:
        if opts.excel_sheet:
            df = pd.read_excel(file_obj, sheet_name=opts.excel_sheet, **_dtype_backend_kwargs(opts))
            return df
        sheets = pd.read_excel(file_obj, sheet_name=None, **_dtype_backend_kwargs(opts))
        if isinstance(sheets, dict):
            non_empty = [(n, d) for n, d in sheets.items() if isinstance(d, pd.DataFrame) and not d.empty]
            if not non_empty:
                return pd.DataFrame()
            _, best = max(non_empty, key=lambda kv: len(kv[1]))
            return best
        return sheets  # type: ignore
    except Exception as e:
        # spróbuj openpyxl/xlrd fallback
        try:
            df = pd.read_excel(file_obj, engine="openpyxl", **_dtype_backend_kwargs(opts))
            return df
        except Exception:
            raise ValueError(f"Nie udało się wczytać XLSX/XLS: {e}")

def _read_json(buf: bytes, opts: ParseOptions) -> pd.DataFrame:
    # Wspiera JSON Lines oraz tablicę obiektów
    # wykryj encoding
    enc = _detect_encoding(buf, opts.csv_try_encodings, opts.csv_sniff_bytes)
    text = buf.decode(enc, errors="ignore").strip()
    # NDJSON?
    if opts.ndjson or (text.startswith("{") and "\n" in text):
        try:
            return pd.read_json(io.StringIO(text), lines=True, **_dtype_backend_kwargs(opts))
        except Exception:
            pass
    # Tablica obiektów
    if text.startswith("[") and text.endswith("]"):
        try:
            arr = json.loads(text)
            if isinstance(arr, list):
                try:
                    return pd.json_normalize(arr, max_level=1)  # płytka normalizacja
                except Exception:
                    return pd.DataFrame(arr)
        except Exception:
            pass
    # Fallback do pandas
    return pd.read_json(io.StringIO(text), **_dtype_backend_kwargs(opts))

def _docx_to_text(file_obj: io.BytesIO) -> str:
    if Document is None:
        return "[DOCX parser niedostępny — zainstaluj python-docx]"
    try:
        doc = Document(file_obj)
        return "\n".join(p.text for p in doc.paragraphs if p is not None)
    except Exception as e:
        return f"[Błąd czytania DOCX: {e}]"

def _pdf_to_text(file_obj: io.BytesIO) -> str:
    if PdfReader is None:
        return "[PDF parser niedostępny — zainstaluj PyPDF2]"
    try:
        reader = PdfReader(file_obj)
    except Exception as e:
        return f"[Błąd otwierania PDF: {e}]"
    parts: List[str] = []
    for i, page in enumerate(reader.pages):
        try:
            parts.append(page.extract_text() or "")
        except Exception:
            parts.append(f"[Błąd ekstrakcji na stronie {i+1}]")
    text = "\n".join(parts).strip()
    return text or "[PDF nie zawiera ekstraktowalnego tekstu]"


# =========================
# Public PRO API
# =========================
def parse_any_pro(file_name: str, file_bytes: bytes, opts: ParseOptions = ParseOptions()) -> ParseResult:
    """
    PRO API: zwraca obiekt z ostrzeżeniami i metadanymi.
    Zgodność: bazowe `parse_any` wykorzystuje to pod spodem.
    """
    if not file_name or file_bytes is None:
        raise ValueError("Brak nazwy pliku lub pusty bufor danych.")

    warnings: List[str] = []
    meta: Dict[str, Any] = {"file_name": file_name, "size_bytes": len(file_bytes)}

    # Rozszerzenie + warianty skompresowane
    base = file_name.lower()
    if base.endswith(".csv.gz"):
        ext = "csv.gz"
    elif base.endswith(".tsv.gz"):
        ext = "tsv.gz"
    else:
        ext = base.split(".")[-1]

    buf = io.BytesIO(file_bytes)

    # Tabelaryczne
    if ext in {"csv", "txt", "csv.gz"}:
        raw = gzip.decompress(file_bytes) if ext == "csv.gz" else file_bytes
        df = _read_csv_like(raw, sep_hint=None, opts=opts)
        df = _maybe_limit(df, opts)
        df = _apply_common_fixes(df, opts)
        meta["format"] = "csv"
        return ParseResult(df=df, text=None, warnings=warnings, meta=meta)

    if ext in {"tsv", "tsv.gz"}:
        raw = gzip.decompress(file_bytes) if ext == "tsv.gz" else file_bytes
        df = _read_csv_like(raw, sep_hint="\t", opts=opts)
        df = _maybe_limit(df, opts)
        df = _apply_common_fixes(df, opts)
        meta["format"] = "tsv"
        return ParseResult(df=df, text=None, warnings=warnings, meta=meta)

    if ext in {"xlsx", "xls"}:
        df = _read_xlsx(buf, opts)
        df = _maybe_limit(df, opts)
        df = _apply_common_fixes(df, opts)
        meta["format"] = "excel"
        return ParseResult(df=df, text=None, warnings=warnings, meta=meta)

    if ext in {"json", "ndjson"}:
        df = _read_json(file_bytes, opts)
        df = _maybe_limit(df, opts)
        df = _apply_common_fixes(df, opts)
        meta["format"] = "jsonl" if ext == "ndjson" else "json"
        return ParseResult(df=df, text=None, warnings=warnings, meta=meta)

    if ext == "parquet":
        try:
            df = pd.read_parquet(buf, **_dtype_backend_kwargs(opts))
        except Exception as e:
            raise ValueError(f"Nie udało się wczytać Parquet: {e}")
        df = _maybe_limit(df, opts)
        df = _apply_common_fixes(df, opts)
        meta["format"] = "parquet"
        return ParseResult(df=df, text=None, warnings=warnings, meta=meta)

    if ext == "feather":
        try:
            df = pd.read_feather(buf, **_dtype_backend_kwargs(opts))
        except Exception as e:
            raise ValueError(f"Nie udało się wczytać Feather: {e}")
        df = _maybe_limit(df, opts)
        df = _apply_common_fixes(df, opts)
        meta["format"] = "feather"
        return ParseResult(df=df, text=None, warnings=warnings, meta=meta)

    # Dokumenty tekstowe
    if ext == "docx":
        meta["format"] = "docx"
        return ParseResult(df=None, text=_docx_to_text(buf), warnings=warnings, meta=meta)

    if ext == "pdf":
        meta["format"] = "pdf"
        return ParseResult(df=None, text=_pdf_to_text(buf), warnings=warnings, meta=meta)

    # Legacy .doc
    if ext == "doc":
        meta["format"] = "doc"
        txt = f"[legacy .doc detected: {len(file_bytes)} bytes] Proszę przekonwertować na .docx."
        return ParseResult(df=None, text=txt, warnings=warnings, meta=meta)

    raise ValueError(f"Unsupported extension: .{ext}")


# =========================
# Public API (back-compat)
# =========================
def parse_any(file_name: str, file_bytes: bytes) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Uniwersalny parser:
    - Zwraca (DataFrame, None) dla formatów tabelarycznych
    - Zwraca (None, str) dla dokumentów tekstowych (DOCX/PDF)
    - Rzuca ValueError dla nieobsługiwanych rozszerzeń
    """
    res = parse_any_pro(file_name, file_bytes, ParseOptions())
    return (res.df if isinstance(res.df, pd.DataFrame) else None, res.text)
