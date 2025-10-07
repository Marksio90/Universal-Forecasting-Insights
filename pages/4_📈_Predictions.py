import io
import json
import time
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

from src.ml_models.automl_pipeline import train_automl
from src.utils.helpers import infer_problem_type

st.title("📈 Predictions (AutoML) — PRO")

# ---------------------------------
# Dane wejściowe
# ---------------------------------
df = st.session_state.get("df")
if df is None or not isinstance(df, pd.DataFrame) or df.empty:
    st.warning("Brak przetworzonych danych (użyj Upload → Szybkie czyszczenie + FE).")
    st.stop()

# ---------------------------------
# Sidebar: Ustawienia treningu
# ---------------------------------
with st.sidebar:
    st.subheader("⚙️ Ustawienia treningu")
    # Podpowiedz cel na bazie heurystyk
    hint_names = ("target", "y", "label", "sales", "sprzeda", "zuzy", "consum", "amount", "profit", "revenue")
    hinted = [c for c in df.columns if any(h in c.lower() for h in hint_names)]
    target = st.selectbox(
        "Kolumna celu",
        options=list(df.columns),
        index=(list(df.columns).index(hinted[0]) if hinted else 0),
        help="Wybierz zmienną, którą model ma przewidywać.",
    )
    st.session_state["target"] = target

    # Parametry (te nie zmieniają działania backendu AutoML, ale używamy ich przy ewaluacji/SHAP)
    test_size = st.slider("Wielkość testu", 0.1, 0.4, 0.2, 0.05)
    random_state = st.number_input("Random state", min_value=0, value=42, step=1)
    enable_shap = st.checkbox("Oblicz SHAP (max 2000 próbek)", value=True)
    shap_max_n = st.number_input("Maks. próbek do SHAP", min_value=200, max_value=5000, value=2000, step=100)
    st.caption("Uwaga: SHAP może zająć trochę czasu dla dużych danych.")

# ---------------------------------
# Detekcja typu problemu
# ---------------------------------
ptype_guess = infer_problem_type(df, target)
if ptype_guess:
    st.caption(f"Sugestia typu problemu: **{ptype_guess}**")

# ---------------------------------
# Trening
# ---------------------------------
if st.button("🚀 Trenuj AutoML", type="primary"):
    t0 = time.time()
    with st.spinner("Trenuję model (AutoML)…"):
        model, metrics, ptype = train_automl(df, target)  # backend wybierze algorytm
    st.session_state["model"] = model
    st.session_state["problem_type"] = ptype

    st.success(f"Wytrenowano model (**{ptype}**) w {time.time()-t0:.2f}s")
    st.json(metrics)

    # ---------------------------------
    # Ewaluacja na holdoucie (spójna z test_size/random_state)
    # ---------------------------------
    y = df[target]
    X = df.drop(columns=[target])

    # Stratyfikacja dla klasyfikacji, jeżeli możliwa
    stratify = y if (ptype == "classification" and y.nunique() > 1) else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
        if stratify is not None and len(y) == len(stratify)
        else None,
    )

    # Predykcje
    try:
        y_pred = model.predict(X_test)
    except Exception as e:
        st.error(f"Nie udało się przewidzieć na holdoucie: {e}")
        st.stop()

    # ---------------------------------
    # Klasyfikacja
    # ---------------------------------
    if ptype == "classification":
        # Raport + macierz
        try:
            report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            st.subheader("📋 Kluczowe metryki (classification)")
            st.json(report)
        except Exception as e:
            st.info(f"Raport klasyfikacji pominięty: {e}")

        cm = confusion_matrix(y_test, y_pred, labels=sorted(y_test.unique()))
        fig_cm = px.imshow(
            cm,
            x=sorted(y_test.unique()),
            y=sorted(y_test.unique()),
            text_auto=True,
            aspect="auto",
            title="Confusion matrix (holdout)",
        )
        st.plotly_chart(fig_cm, use_container_width=True)

        # ROC AUC (jeśli dostępne predict_proba i binarka/multiclass)
        if hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba(X_test)
                if proba.ndim == 2 and proba.shape[1] == 2:
                    auc = roc_auc_score(y_test, proba[:, 1])
                    fpr, tpr, _ = roc_curve(y_test, proba[:, 1])
                    st.metric("ROC AUC", f"{auc:.4f}")
                    st.plotly_chart(
                        go.Figure(
                            data=[go.Scatter(x=fpr, y=tpr, mode="lines", name="ROC")],
                            layout=go.Layout(title="ROC Curve", xaxis_title="FPR", yaxis_title="TPR"),
                        ),
                        use_container_width=True,
                    )
            except Exception as e:
                st.caption(f"ROC AUC pominięty: {e}")

    # ---------------------------------
    # Regresja
    # ---------------------------------
    elif ptype == "regression":
        rmse = float(mean_squared_error(y_test, y_pred, squared=False))
        mae = float(mean_absolute_error(y_test, y_pred))
        r2 = float(r2_score(y_test, y_pred))
        st.subheader("📋 Kluczowe metryki (regression)")
        st.json({"rmse": rmse, "mae": mae, "r2": r2})

        # Parity plot
        df_par = pd.DataFrame({"y_true": y_test.values, "y_pred": y_pred})
        fig_par = px.scatter(df_par, x="y_true", y="y_pred", title="Parity plot (y_true vs y_pred)", trendline="ols")
        st.plotly_chart(fig_par, use_container_width=True)

        # Residuals
        df_res = pd.DataFrame({"y_true": y_test.values, "resid": y_test.values - y_pred})
        fig_res = px.histogram(df_res, x="resid", nbins=40, title="Rozkład reszt (y_true - y_pred)")
        st.plotly_chart(fig_res, use_container_width=True)

    # ---------------------------------
    # Feature importance (jeśli dostępne)
    # ---------------------------------
    st.subheader("🏷️ Ważność cech")
    if hasattr(model, "feature_importances_"):
        try:
            imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False).head(30)
            st.bar_chart(imp)
        except Exception as e:
            st.caption(f"Wizualizacja ważności pominięta: {e}")
    else:
        st.caption("Model nie udostępnia atrybutu `feature_importances_`.")

    # ---------------------------------
    # SHAP (opcjonalnie)
    # ---------------------------------
    if enable_shap:
        try:
            import shap  # noqa: F401

            st.subheader("🧠 SHAP summary (próbka)")
            # próbkowanie do SHAP
            nsamp = min(len(X_test), shap_max_n)
            X_shap = X_test.sample(nsamp, random_state=random_state) if len(X_test) > nsamp else X_test

            # TreeExplainer dla drzew, fallback do KernelExplainer (ostrożnie)
            explainer = None
            if model.__class__.__name__.lower().startswith(("lgbm", "xgb", "randomforest", "gradientboosting")):
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_shap)
            else:
                # KernelExplainer może być kosztowny; dlatego bardzo mały sampling
                X_kernel = X_shap.sample(min(len(X_shap), 400), random_state=random_state)
                explainer = shap.KernelExplainer(model.predict, X_kernel)
                shap_values = explainer.shap_values(X_kernel)

            # Rysowanie do HTML (użyj komponentu shap.plots)
            st.caption("Uwaga: wykres generowany po stronie serwera (może potrwać chwilę).")
            shap_html = io.StringIO()
            shap.plots.beeswarm(shap_values, max_display=20, show=False)
            # Matplotlib → PNG → pokaz w Streamlit
            import matplotlib.pyplot as plt
            buf = io.BytesIO()
            plt.tight_layout()
            plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
            plt.close()
            st.image(buf.getvalue(), caption="SHAP summary (próbka)", use_container_width=True)
        except Exception as e:
            st.info(f"SHAP pominięty: {e}")

    # ---------------------------------
    # Eksport: model, metryki, predykcje holdout
    # ---------------------------------
    st.subheader("⬇️ Eksport artefaktów")
    # model
    try:
        model_bytes = io.BytesIO()
        joblib.dump(model, model_bytes)
        model_bytes.seek(0)
        st.download_button(
            "Pobierz model (.joblib)",
            data=model_bytes,
            file_name=f"model_{ptype}.joblib",
            mime="application/octet-stream",
        )
    except Exception as e:
        st.caption(f"Nie udało się wyeksportować modelu: {e}")

    # metryki
    try:
        metrics_export = {"automl_metrics": metrics, "ptype": ptype}
        if ptype == "regression":
            metrics_export.update({"rmse": rmse, "mae": mae, "r2": r2})
        metrics_json = io.StringIO()
        json.dump(metrics_export, metrics_json, ensure_ascii=False, indent=2)
        st.download_button(
            "Pobierz metryki (.json)",
            data=metrics_json.getvalue(),
            file_name="metrics.json",
            mime="application/json",
        )
    except Exception as e:
        st.caption(f"Nie udało się wyeksportować metryk: {e}")

    # predykcje holdout
    try:
        preds_df = pd.DataFrame({"y_true": y_test}).reset_index(drop=True)
        preds_df["y_pred"] = y_pred if np.ndim(y_pred) == 1 else np.argmax(y_pred, axis=1)
        preds_csv = io.StringIO()
        preds_df.to_csv(preds_csv, index=False)
        st.download_button(
            "Pobierz predykcje holdout (.csv)",
            data=preds_csv.getvalue(),
            file_name="predictions_holdout.csv",
            mime="text/csv",
        )
    except Exception as e:
        st.caption(f"Nie udało się wyeksportować predykcji: {e}")
else:
    st.caption("Ustaw parametry i uruchom trening.")
