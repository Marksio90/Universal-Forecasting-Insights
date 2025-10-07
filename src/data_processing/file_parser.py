from __future__ import annotations
import io
import csv
import json
from typing import Tuple, Optional

import pandas as pd
from PyPDF2 import PdfReader
from docx import Document

# Rozszerzony zestaw (zachowujemy kompatybilność z wcześniejszą listą)
SUPPORTED = {"csv", "tsv", "txt", "xlsx", "json", "docx", "pdf"}  # .doc -> best-effort komunikat

# -----------------------------
# Helpers
# -----------------------------
_ENCODING_CANDIDATES = ("utf-8", "utf-8-sig", "cp1250", "iso-8859-2", "latin-1")

def _try_read_csv_with_encodings(buf0: bytes, sep_hint: Optional[str] = None) -> pd.DataFrame:
    """
    Próbuj wczytać CSV/TSV przy różnych kodowaniach i heurystycznie dobranym separatorze.
    """
    # 1) wykryj separator (jeśli nie podano)
    if sep_hint is None:
        sample = buf0[:10000].decode("utf-8", errors="ignore")
        # szybka heurystyka: jeśli dużo tabów -> TSV
        if sample.count("\t") > sample.count(",") and sample.count("\t") >= 2:
            sep_hint = "\t"
        else:
            # csv.Sniffer może rzucić wyjątek; fallback na przecinek/średnik
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=[",", ";", "\t", "|"])
                sep_hint = dialect.delimiter
            except Exception:
                # jeżeli dużo średników – weź średnik
                sep_hint = ";" if sample.count(";") > sample.count(",") else ","

    # 2) próby kodowań
    for enc in _ENCODING_CANDIDATES:
        try:
            return pd.read_csv(io.BytesIO(buf0), sep=sep_hint, encoding=enc, low_memory=False)
        except Exception:
            continue

    # 3) ostateczny fallback – spróbuj bez określania sep (pandas wykryje)
    try:
        return pd.read_csv(io.BytesIO(buf0), low_memory=False)
    except Exception as e:
        raise ValueError(f"Nie udało się wczytać CSV/TSV: {e}")

def _read_csv(file) -> pd.DataFrame:
    data = file.read() if hasattr(file, "read") else file
    if not isinstance(data, (bytes, bytearray)):
        raise ValueError("Nieprawidłowy bufor CSV.")
    return _try_read_csv_with_encodings(data)

def _read_tsv_or_txt(file) -> pd.DataFrame:
    data = file.read() if hasattr(file, "read") else file
    if not isinstance(data, (bytes, bytearray)):
        raise ValueError("Nieprawidłowy bufor TSV/TXT.")
    return _try_read_csv_with_encodings(data, sep_hint="\t")

def _read_xlsx(file) -> pd.DataFrame:
    """
    Wczytuje XLSX. Jeśli jest wiele arkuszy, wybiera największy niepusty.
    """
    try:
        # sheet_name=None -> dict arkuszy
        xls = pd.read_excel(file, sheet_name=None)
        if isinstance(xls, dict):
            # wybierz największy niepusty arkusz
            non_empty = [(name, df) for name, df in xls.items() if isinstance(df, pd.DataFrame) and not df.empty]
            if not non_empty:
                return pd.DataFrame()
            name, best = max(non_empty, key=lambda kv: len(kv[1]))
            return best
        # w przypadku pojedynczego arkusza
        return xls
    except Exception as e:
        raise ValueError(f"Nie udało się wczytać XLSX: {e}")

def _read_json(file) -> pd.DataFrame:
    """
    Obsługuje:
      - JSON array of objects -> DataFrame
      - JSON Lines (NDJSON) -> lines=True
      - Fallback: pandas.read_json
    """
    raw = file.read() if hasattr(file, "read") else file
    if not isinstance(raw, (bytes, bytearray)):
        raise ValueError("Nieprawidłowy bufor JSON.")
    # spróbuj dekodować w popularnych kodowaniach
    text = None
    for enc in _ENCODING_CANDIDATES:
        try:
            text = raw.decode(enc)
            break
        except Exception:
            continue
    if text is None:
        text = raw.decode("utf-8", errors="ignore")

    stripped = text.strip()

    # JSON Lines?
    if "\n" in stripped and stripped.lstrip().startswith("{"):
        try:
            return pd.read_json(io.StringIO(stripped), lines=True)
        except Exception:
            pass

    # Tablica obiektów?
    if stripped.startswith("[") and stripped.endswith("]"):
        try:
            arr = json.loads(stripped)
            if isinstance(arr, list):
                # Spróbuj znormalizować (dla zagnieżdżeń)
                try:
                    return pd.json_normalize(arr, max_level=1)
                except Exception:
                    return pd.DataFrame(arr)
        except Exception:
            pass

    # Fallback: pandas.read_json
    try:
        return pd.read_json(io.StringIO(stripped))
    except Exception as e:
        raise ValueError(f"Nie udało się wczytać JSON: {e}")

def _docx_to_text(file) -> str:
    try:
        doc = Document(file)
        return "\n".join(p.text for p in doc.paragraphs if p is not None)
    except Exception as e:
        return f"[Błąd czytania DOCX: {e}]"

def _pdf_to_text(file) -> str:
    try:
        reader = PdfReader(file)
    except Exception as e:
        return f"[Błąd otwierania PDF: {e}]"
    parts = []
    for i, page in enumerate(reader.pages):
        try:
            parts.append(page.extract_text() or "")
        except Exception:
            parts.append(f"[Błąd ekstrakcji na stronie {i+1}]")
    text = "\n".join(parts).strip()
    return text or "[PDF nie zawiera ekstraktowalnego tekstu]"

# -----------------------------
# Public API
# -----------------------------
def parse_any(file_name: str, file_bytes: bytes) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Uniwersalny parser:
    - Zwraca (DataFrame, None) dla formatów tabelarycznych
    - Zwraca (None, str) dla dokumentów tekstowych (DOCX/PDF)
    - Rzuca ValueError dla nieobsługiwanych rozszerzeń

    Uwaga: zachowujemy kontrakt zwrotu 2-elementowej krotki.
    """
    if not file_name or file_bytes is None:
        raise ValueError("Brak nazwy pliku lub pusty bufor danych.")

    ext = file_name.split(".")[-1].lower()
    buf = io.BytesIO(file_bytes)

    if ext not in SUPPORTED and ext != "doc":
        raise ValueError(f"Unsupported extension: .{ext}")

    # Tabelaryczne
    if ext in {"csv", "txt"}:
        df = _read_csv(buf)
        return (df if not df.empty else pd.DataFrame(), None)

    if ext == "tsv":
        df = _read_tsv_or_txt(buf)
        return (df if not df.empty else pd.DataFrame(), None)

    if ext == "xlsx":
        df = _read_xlsx(buf)
        return (df if not df.empty else pd.DataFrame(), None)

    if ext == "json":
        df = _read_json(buf)
        return (df if not df.empty else pd.DataFrame(), None)

    # Dokumenty tekstowe
    if ext == "docx":
        return None, _docx_to_text(buf)

    if ext == "pdf":
        return None, _pdf_to_text(buf)

    # Legacy .doc
    if ext == "doc":
        return None, f"[legacy .doc detected: {len(file_bytes)} bytes] Proszę przekonwertować na .docx."

    # Default (nie powinno zajść z racji SUPPORTED)
    raise ValueError(f"Unsupported extension: .{ext}")
