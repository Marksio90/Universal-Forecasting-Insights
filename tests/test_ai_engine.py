# tests/test_ai_engine.py
from __future__ import annotations
import types
import json
import pytest
import pandas as pd

# =========================================================
# openai_integrator.chat_completion
# =========================================================
def test_chat_completion_returns_error_without_key(monkeypatch):
    # brak klucza -> get_client() zwróci None
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    from src.ai_engine import openai_integrator as oi

    out = oi.chat_completion(system="sys", user="hello")
    assert isinstance(out, str)
    assert "Brak klucza OpenAI" in out


def test_chat_completion_success_with_mocked_client(monkeypatch):
    # ustaw "klucz", ale podstawiamy fałszywego klienta OpenAI
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    from src.ai_engine import openai_integrator as oi

    # Fake response structure zgodna z odczytem w kodzie (choices[0].message.content)
    class _Msg: 
        def __init__(self, content): self.content = content
    class _Choice:
        def __init__(self, content): self.message = _Msg(content)
    class _Resp:
        def __init__(self, content): self.choices = [_Choice(content)]

    calls = {}

    class _Completions:
        def create(self, model, messages, temperature, max_tokens):
            calls["model"] = model
            calls["messages"] = messages
            calls["temperature"] = temperature
            calls["max_tokens"] = max_tokens
            return _Resp("MOCK-RESPONSE")

    class _Chat:
        def __init__(self): self.completions = _Completions()

    class _Client:
        def __init__(self, api_key=None): self.chat = _Chat()

    # podmień klasę OpenAI używaną w get_client()
    monkeypatch.setattr(oi, "OpenAI", lambda api_key=None: _Client(api_key=api_key))

    out = oi.chat_completion(system="sys", user="hello", model="gpt-4o-mini", temperature=0.2, max_tokens=123)
    assert out == "MOCK-RESPONSE"
    # sanity check parametrów
    assert calls["model"] == "gpt-4o-mini"
    assert calls["temperature"] == 0.2
    assert calls["max_tokens"] == 123
    assert calls["messages"][0]["role"] == "system"
    assert calls["messages"][1]["role"] == "user"


def test_chat_completion_handles_exception(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    from src.ai_engine import openai_integrator as oi

    class _Boom:
        def create(self, *a, **kw):  # rzuca wyjątek symulując błąd sieci/serwisu
            raise RuntimeError("boom")

    class _Client:
        def __init__(self): 
            self.chat = types.SimpleNamespace(completions=_Boom())

    monkeypatch.setattr(oi, "OpenAI", lambda api_key=None: _Client())
    out = oi.chat_completion(system="sys", user="hello")
    assert out.startswith("Błąd OpenAI:")
    assert "boom" in out


# =========================================================
# insights_generator.generate_insights
# =========================================================
def test_generate_insights_builds_expected_prompt(monkeypatch):
    import src.ai_engine.insights_generator as ig

    captured = {}
    def fake_chat(system: str, user: str, **kw) -> str:
        captured["system"] = system
        captured["user"] = user
        captured["kwargs"] = kw
        return "INSIGHTS"

    monkeypatch.setattr(ig, "chat_completion", fake_chat, raising=True)

    df = pd.DataFrame(
        {
            "a": [1, 2, None, 4],              # 1 NaN
            "b": ["x", "y", "z", None],        # 1 NaN
            "date": pd.date_range("2024-01-01", periods=4, freq="D"),
        }
    )
    txt = ig.generate_insights(df, goal="prognoza sprzedaży Q4")
    assert txt == "INSIGHTS"

    # Sprawdź, że prompt zawiera statystyki
    u = captured["user"]
    assert "wiersze: 4" in u
    assert "kolumny: 3" in u
    assert "braki (%): 25.00%" in u or "braki (%): 25%" in u
    assert "duplikaty:" in u
    assert "Cel użytkownika: prognoza sprzedaży Q4" in u


def test_generate_insights_handles_none_goal(monkeypatch):
    import src.ai_engine.insights_generator as ig

    monkeypatch.setattr(ig, "chat_completion", lambda **kw: "OK")
    df = pd.DataFrame({"x": [1, 2, 3]})
    out = ig.generate_insights(df, goal=None)
    assert out == "OK"


# =========================================================
# report_generator.build_report_html
# =========================================================
def test_build_report_html_renders_jinja_template(tmp_path, monkeypatch):
    from src.ai_engine import report_generator as rg

    # przygotuj tymczasowy szablon
    tpl = tmp_path / "report_template.html"
    tpl.write_text(
        """<!doctype html>
<html><head><meta charset="utf-8"><title>{{ title }}</title></head>
<body>
<h1>{{ title }}</h1>
<div id="m">{{ metrics | tojson }}</div>
{% if notes %}<div id="n">{{ notes }}</div>{% endif %}
</body></html>""",
        encoding="utf-8",
    )

    # podmień ścieżkę do assets w module
    monkeypatch.setattr(rg, "ASSETS", tpl, raising=True)

    ctx = {"title": "Raport Biznesowy", "metrics": {"rmse": 1.23}, "notes": "Notatka"}
    html = rg.build_report_html(ctx)

    assert "Raport Biznesowy" in html
    assert '"rmse": 1.23' in html
    assert "Notatka" in html
    assert html.strip().startswith("<!doctype html>")


# =========================================================
# Drobne sanity dla kompatybilności API
# =========================================================
def test_public_api_shapes():
    # Ten test upewnia się, że importy modułów istnieją i mają oczekiwane symbole
    import src.ai_engine.insights_generator as ig
    import src.ai_engine.report_generator as rg
    import src.ai_engine.openai_integrator as oi

    assert callable(ig.generate_insights)
    assert callable(rg.build_report_html)
    assert callable(oi.chat_completion)
