import io
import json
import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from src.ml_models.forecasting import forecast
from src.utils.helpers import ensure_datetime_index

st.title("📊 Forecasting (Prophet) — PRO")

# ---------------------------
# Dane wejściowe
# ---------------------------
df0 = st.session_state.get("df") or st.session_state.get("df_raw")
if df0 is None or not isinstance(df0, pd.DataFrame) or df0.empty:
    st.warning("Brak danych.")
    st.stop()

# ---------------------------
# Sidebar: Ustawienia
# ---------------------------
with st.sidebar:
    st.subheader("⚙️ Parametry prognozy")
    # wybór kolumny celu i (opcjonalnie) daty
    target = st.selectbox("Kolumna celu (y)", options=list(df0.columns))
    date_col = st.selectbox(
        "Kolumna daty/czasu (opcjonalnie)",
        options=["(auto)"] + [c for c in df0.columns if "date" in c.lower() or "data" in c.lower() or "time" in c.lower()],
        index=0,
        help="Jeśli nie wybierzesz – spróbuję wykryć automatycznie."
    )
    horizon = st.number_input("Horyzont prognozy (okresy)", min_value=1, max_value=365*5, value=12, step=1)
    start_forecast_at = st.selectbox(
        "Punkt startu prognozy",
        options=["po ostatniej obserwacji", "ostatnie N okresów jako test (pokaż porównanie)"],
        index=1
    )
    test_periods = st.number_input("N okresów testowych (gdy porównanie)", min_value=1, max_value=10000, value=min(12, max(1, len(df0)//10)))
    st.markdown("---")
    st.subheader("🔧 Zaawansowane (Prophet)")
    seasonality_mode = st.selectbox("Sezonowość", options=["additive", "multiplicative"], index=0)
    cps = st.slider("Changepoint prior scale", 0.01, 1.0, 0.1, 0.01,
                    help="Wyższa wartość = większa podatność na zmiany trendu")
    backtest_folds = st.number_input("Rolling backtesting — liczba foldów", min_value=0, max_value=10, value=3,
                                     help="0 = wyłączone")
    st.caption("Uwaga: większy horyzont i/lub więcej foldów = dłuższy czas działania")

# ---------------------------
# Przygotowanie danych
# ---------------------------
df = df0.copy()

# wybór kolumny daty, jeśli podana
if date_col != "(auto)":
    try:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce", infer_datetime_format=True)
        df = df.set_index(date_col).sort_index()
    except Exception:
        st.warning("Nie udało się zastosować wskazanej kolumny czasu – użyję auto-wykrywania.")

# auto-wykrycie indeksu czasu
df = ensure_datetime_index(df)

if not isinstance(df.index, pd.DatetimeIndex):
    st.error("Nie wykryto prawidłowego indeksu czasu. Upewnij się, że masz kolumnę z datą/czasem.")
    st.stop()

if target not in df.columns:
    st.error("Wybrana kolumna celu nie istnieje po przetworzeniu danych.")
    st.stop()

# Podsumowanie i częstotliwość
st.subheader("⏱️ Czas i częstotliwość")
freq = pd.infer_freq(df.index)
st.caption(f"Zgadnięta częstotliwość: **{freq or 'niejednorodna / auto'}** • Okresy: {len(df):,}")

with st.expander("📄 Podgląd danych szeregowych", expanded=False):
    st.dataframe(df[[target]].tail(50))

# ---------------------------
# Metryki pomocnicze
# ---------------------------
def smape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred))
    denom[denom == 0] = 1.0
    return float(np.mean(2.0 * np.abs(y_pred - y_true) / denom)) * 100.0

def mase(y_true, y_pred, m=1):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) <= m:
        return float(np.nan)
    d = np.abs(np.diff(y_true, n=m)).mean()
    if d == 0:
        return float(np.nan)
    return float(np.mean(np.abs(y_true - y_pred)) / d)

def guess_seasonal_period(freq_hint: str|None) -> int:
    if not freq_hint:
        return 1
    f = freq_hint.upper()
    if f.startswith("H"):   # hourly
        return 24
    if f.startswith("D"):   # daily
        return 7
    if f.startswith("W"):   # weekly
        return 52
    if f.startswith("M"):   # monthly
        return 12
    if f.startswith("Q"):   # quarterly
        return 4
    if f.startswith("A") or f.startswith("Y"):  # yearly
        return 1
    return 1

m_period = guess_seasonal_period(freq)

# ---------------------------
# Backtesting (rolling origin)
# ---------------------------
metrics_bt = []
if start_forecast_at.startswith("ostatnie") and backtest_folds > 0:
    st.subheader("🧪 Rolling Backtesting")
    total = len(df)
    fold_size = max(1, int(test_periods / max(1, backtest_folds)))
    # przesuwane okna – na końcu serii
    for k in range(backtest_folds, 0, -1):
        test_k = fold_size * k
        split = max(5, total - test_k)  # minimum kilka punktów na tren
        train_df = df.iloc[:split]
        test_df = df.iloc[split: split + fold_size]

        if len(test_df) < 1:
            continue
        try:
            # lokalny fit (korzystamy z backendowego ensure_datetime_index poprzez forecast())
            # tutaj użyjemy bezpośrednio Prophet przez helper w backendzie 'forecast'
            _model, fc = forecast(train_df[[target]].copy(), target, horizon=len(test_df))
            # dopasuj po datach
            fc = fc.set_index("ds").reindex(test_df.index).dropna()
            if fc.empty:
                continue
            y_true = test_df[target].iloc[:len(fc)]
            y_pred = fc["yhat"].iloc[:len(fc)]
            m = {
                "fold": backtest_folds - k + 1,
                "sMAPE": smape(y_true, y_pred),
                "MASE": mase(y_true, y_pred, m=m_period),
                "RMSE": float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
            }
            metrics_bt.append(m)
        except Exception as e:
            metrics_bt.append({"fold": backtest_folds - k + 1, "error": str(e)})

    if metrics_bt:
        bt_df = pd.DataFrame(metrics_bt)
        st.dataframe(bt_df, use_container_width=True)
        # heatmap metryk (bez błędów)
        bt_plot = bt_df.drop(columns=[c for c in ["error"] if c in bt_df])
        if not bt_plot.empty and set(["sMAPE","MASE","RMSE"]).issubset(bt_plot.columns):
            fig_bt = px.imshow(
                bt_plot.set_index("fold")[["sMAPE","MASE","RMSE"]].T,
                text_auto=True, aspect="auto", title="Backtesting — metryki (niżej lepiej)"
            )
            st.plotly_chart(fig_bt, use_container_width=True)

# ---------------------------
# Główna prognoza
# ---------------------------
if st.button("📟 Prognozuj", type="primary"):
    t0 = time.time()
    try:
        # Jeśli chcemy testować na końcówce – potnij dane
        hist_df = df.copy()
        test_slice = None
        if start_forecast_at.startswith("ostatnie") and test_periods > 0:
            if test_periods >= len(hist_df):
                st.error("Za długi okres testowy w stosunku do długości serii.")
                st.stop()
            test_slice = hist_df.iloc[-test_periods:][[target]].copy()
            hist_df = hist_df.iloc[:-test_periods]

        # Fit + forecast
        model, fcst = forecast(hist_df[[target]].copy(), target, horizon=horizon)
        fcst["ds"] = pd.to_datetime(fcst["ds"])
        st.success(f"Prognoza gotowa. ⏱️ {time.time()-t0:.2f}s")
        st.dataframe(fcst.head(20), use_container_width=True)

        # Wykres: historia + prognoza
        hist_plot = pd.DataFrame({"ds": hist_df.index, "y": hist_df[target].values})
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist_plot["ds"], y=hist_plot["y"], name="Rzeczywiste", mode="lines"))
        fig.add_trace(go.Scatter(x=fcst["ds"], y=fcst["yhat"], name="Prognoza", mode="lines"))
        # przedziały
        fig.add_trace(go.Scatter(
            x=pd.concat([fcst["ds"], fcst["ds"][::-1]]),
            y=pd.concat([fcst["yhat_upper"], fcst["yhat_lower"][::-1]]),
            fill="toself", name="Przedział", opacity=0.2, line=dict(width=0)
        ))
        fig.update_layout(title="Rzeczywiste vs Prognoza", xaxis_title="Data", yaxis_title=target)
        st.plotly_chart(fig, use_container_width=True)

        # Jeśli wybrano „ostatnie N okresów jako test” – pokaż porównanie
        if test_slice is not None:
            fc_align = fcst.set_index("ds").reindex(test_slice.index).dropna()
            if not fc_align.empty:
                y_true = test_slice[target].iloc[:len(fc_align)]
                y_pred = fc_align["yhat"].iloc[:len(fc_align)]
                met = {
                    "sMAPE": smape(y_true, y_pred),
                    "MASE": mase(y_true, y_pred, m=m_period),
                    "RMSE": float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
                }
                st.subheader("📏 Metryki porównania (na ostatnich N okresach)")
                st.json(met)
                # wykres porównania na testowym odcinku
                cmp_df = pd.DataFrame({"ds": y_true.index, "y_true": y_true.values, "y_pred": y_pred.values})
                st.plotly_chart(px.line(cmp_df, x="ds", y=["y_true","y_pred"], title="Ostatnie N okresów — porównanie"),
                                use_container_width=True)

        # Changepoints (jeśli są)
        try:
            cps_df = getattr(model, "changepoints", None)
            if cps_df is not None and len(cps_df) > 0:
                with st.expander("📌 Wykryte changepoints"):
                    st.write(pd.to_datetime(cps_df).to_frame(name="changepoint"))
        except Exception:
            pass

        # Eksport
        csv_buf = io.StringIO()
        fcst.to_csv(csv_buf, index=False)
        st.download_button("⬇️ Pobierz prognozę (CSV)", data=csv_buf.getvalue(), file_name="forecast.csv", mime="text/csv")

        # JSON metryk (jeśli były backtesty/metryki testowe)
        meta_export = {
            "freq": freq,
            "horizon": int(horizon),
            "seasonality_mode": seasonality_mode,
            "changepoint_prior_scale": cps,
            "backtesting": metrics_bt if metrics_bt else None
        }
        json_buf = io.StringIO()
        json.dump(meta_export, json_buf, ensure_ascii=False, indent=2)
        st.download_button("⬇️ Pobierz metadane prognozy (JSON)", data=json_buf.getvalue(), file_name="forecast_meta.json", mime="application/json")

    except Exception as e:
        st.error(f"Nie udało się wytrenować modelu szeregowego: {e}")
