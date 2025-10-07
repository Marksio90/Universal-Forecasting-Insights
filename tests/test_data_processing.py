# tests/test_data_processing.py
from __future__ import annotations
import io
import json
import math
import pytest
import pandas as pd
import numpy as np

# ===============================
# file_parser.parse_any
# ===============================
def _csv_bytes(text: str) -> bytes:
    return text.encode("utf-8")

def _xlsx_bytes(df: pd.DataFrame) -> bytes:
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as xw:
        df.to_excel(xw, index=False)
    return bio.getvalue()

def _json_bytes(obj) -> bytes:
    return json.dumps(obj).encode("utf-8")

def _docx_bytes(text: str) -> bytes:
    try:
        from docx import Document  # python-docx
    except Exception:  # pragma: no cover
        pytest.skip("python-docx not available")
    bio = io.BytesIO()
    d = Document()
    d.add_paragraph(text)
    d.save(bio)
    return bio.getvalue()

def _pdf_bytes() -> bytes:
    # Tworzymy minimalny PDF (pusta strona). Ekstrakcja tekstu może dać "" — i to jest OK.
    try:
        from PyPDF2 import PdfWriter
    except Exception:  # pragma: no cover
        pytest.skip("PyPDF2 not available")
    bio = io.BytesIO()
    writer = PdfWriter()
    writer.add_blank_page(width=300, height=300)
    writer.write(bio)
    return bio.getvalue()

def test_parse_any_csv_basic():
    from src.data_processing.file_parser import parse_any
    data = "a,b\n1,2\n3,4\n"
    df, txt = parse_any("test.csv", _csv_bytes(data))
    assert txt is None
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["a", "b"]
    assert df.shape == (2, 2)

def test_parse_any_xlsx_roundtrip():
    from src.data_processing.file_parser import parse_any
    df0 = pd.DataFrame({"x": [1, 2], "y": [3.5, 4.5]})
    df, txt = parse_any("test.xlsx", _xlsx_bytes(df0))
    assert txt is None
    pd.testing.assert_frame_equal(df.reset_index(drop=True), df0.reset_index(drop=True))

def test_parse_any_json_records():
    from src.data_processing.file_parser import parse_any
    obj = [{"a": 1}, {"a": 2}]
    df, txt = parse_any("data.json", _json_bytes(obj))
    assert txt is None
    assert df["a"].tolist() == [1, 2]

def test_parse_any_docx_text():
    from src.data_processing.file_parser import parse_any
    df, txt = parse_any("doc.docx", _docx_bytes("Hello world\nLinia 2"))
    assert df is None
    assert isinstance(txt, str)
    assert "Hello world" in txt

def test_parse_any_pdf_graceful():
    from src.data_processing.file_parser import parse_any
    df, txt = parse_any("x.pdf", _pdf_bytes())
    assert df is None
    assert isinstance(txt, str)  # może być pusty string, ważne że nie wyjątek

def test_parse_any_legacy_doc_message():
    from src.data_processing.file_parser import parse_any
    df, txt = parse_any("legacy.doc", b"binary")
    assert df is None
    assert isinstance(txt, str)
    assert "legacy .doc detected" in txt.lower()

def test_parse_any_unsupported():
    from src.data_processing.file_parser import parse_any
    with pytest.raises(ValueError):
        parse_any("image.png", b"...")

# ===============================
# data_cleaner.quick_clean
# ===============================
def test_quick_clean_basic_behaviour():
    from src.data_processing.data_cleaner import quick_clean
    df = pd.DataFrame(
        {
            "id": [1, 1, 2, 3],
            "price": [" 1 234,50 ", "1 234,50", "2 000,00", None],
            "txt": [" A ", "B ", None, "C"],
            "num": [1, np.nan, np.inf, -np.inf],
        }
    )
    out = quick_clean(df)

    # duplikaty out (drugi wiersz to duplikat id? quick_clean usuwa duplikaty całych wierszy, więc kształt >= 3)
    assert len(out) <= len(df)

    # price został przekonwertowany do liczby i uzupełniony medianą
    assert pd.api.types.is_numeric_dtype(out["price"])
    assert out["price"].isna().sum() == 0

    # stringi przycięte
    assert out["txt"].dropna().str.startswith(" ").sum() == 0
    assert out["txt"].dropna().str.endswith(" ").sum() == 0

    # inf → NaN → median fill
    assert np.isfinite(out["num"]).all()

# ===============================
# feature_engineering.basic_feature_engineering
# ===============================
def test_basic_feature_engineering_dates_and_cats():
    from src.data_processing.feature_engineering import basic_feature_engineering
    df = pd.DataFrame(
        {
            "OrderDate": ["2024-01-01", "2024-02-02", "2024-03-03", "bad"],
            "Segment": ["A", "B", "A", "C"],
            "value": [10, 20, 30, 40],
        }
    )
    out = basic_feature_engineering(df)

    # powstaną kolumny date_* (>= jedna z: year/month/dow/day)
    date_cols = [c for c in out.columns if c.lower().startswith("orderdate_")]
    assert len(date_cols) >= 3

    # Segment o niskiej krotności zakodowany numerycznie
    assert pd.api.types.is_integer_dtype(out["Segment"])

# ===============================
# data_validator.validate (używa utils.validators.basic_quality_checks)
# ===============================
def test_data_validator_validate_report():
    from src.data_processing.data_validator import validate
    df = pd.DataFrame({"a": [1, None, 3], "b": ["x", "y", "y"]})
    rep = validate(df)
    # minimalne pola (z Twojego PRO-validators)
    assert rep["rows"] == 3
    assert rep["cols"] == 2
    assert "missing_pct" in rep
    assert "dupes" in rep
    # pola rozszerzone (jeśli PRO validator wgrany)
    assert "dtypes_summary" in rep
    assert "cardinality" in rep

# ===============================
# data_profiler.make_profile_html
# ===============================
def test_data_profiler_html_smoke(monkeypatch):
    yp = pytest.importorskip("ydata_profiling")
    from src.data_processing.data_profiler import make_profile_html
    df = pd.DataFrame({"a": [1, 2, 3, 4], "b": [1, 1, 2, 2]})
    html = make_profile_html(df, title="Test Profile")
    assert isinstance(html, str)
    assert "<html" in html.lower()

# ===============================
# utils.helpers – smart_cast_numeric / ensure_datetime_index / infer_problem_type
# ===============================
def test_smart_cast_numeric_handles_pl_formats():
    from src.utils.helpers import smart_cast_numeric
    df = pd.DataFrame(
        {
            "pln": ["1 234,56 zł", "2 000,00", None, "3,50"],
            "pct": ["10%", "5 %", "0,5%", None],
            "txt": ["tak", "nie", "YES", "OFF"],  # bool-like → może zamienić się w 0/1
        }
    )
    out = smart_cast_numeric(df)
    # waluty na float
    assert pd.api.types.is_numeric_dtype(out["pln"])
    assert math.isclose(float(out["pln"].dropna().iloc[0]), 1234.56, rel_tol=1e-6)
    # procenty na [0..1]
    assert pd.api.types.is_numeric_dtype(out["pct"])
    assert 0 <= out["pct"].dropna().iloc[0] <= 1

def test_ensure_datetime_index_and_problem_type():
    from src.utils.helpers import ensure_datetime_index, infer_problem_type
    raw = pd.DataFrame(
        {
            "Data": ["01-01-2024", "02-01-2024", "03-01-2024"],
            "y": [1.0, 2.0, 3.0],
        }
    )
    out = ensure_datetime_index(raw)
    assert isinstance(out.index, pd.DatetimeIndex)
    # infer → timeseries
    ptype = infer_problem_type(out.reset_index().rename(columns={"index": "ds"}), target="y")
    assert ptype == "timeseries"
