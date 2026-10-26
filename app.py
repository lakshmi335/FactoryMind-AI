"""
FactoryMind AI - Intelligent Predictive Maintenance & Production Optimization Platform
=========================================================================================
Main Streamlit application. Run with:  streamlit run app.py
"""

import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from preprocessing import (
    FAILURE_FEATURES, PRODUCTION_FEATURES, ANOMALY_FEATURES,
    load_machine_data, load_production_data,
)
from failure_model import load_failure_model, predict_failure, risk_category
from production_model import load_production_model, predict_production
from anomaly_model import load_anomaly_model, score_sequence
from forecast_model import load_forecast_model, forecast_next, WINDOW
from explainability import explain_prediction, global_feature_importance
from recommendations import compute_health_score, root_cause_analysis, generate_recommendations

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

st.set_page_config(
    page_title="FactoryMind AI",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# THEME / STYLING
# ----------------------------------------------------------------------------
st.markdown("""
<style>
    .stApp { background-color: #0e1117; }
    div[data-testid="stMetric"] {
        background: linear-gradient(145deg, #1a1f2b, #151922);
        border: 1px solid #2a3040;
        border-radius: 12px;
        padding: 14px 16px;
    }
    div[data-testid="stMetricLabel"] { color: #8b93a7; }
    .badge {
        display: inline-block; padding: 3px 12px; border-radius: 20px;
        font-size: 0.78rem; font-weight: 600; letter-spacing: 0.3px;
    }
    .badge-healthy { background: #10381f; color: #4ade80; border: 1px solid #1f5c34; }
    .badge-warning { background: #3d3010; color: #facc15; border: 1px solid #5c4a1a; }
    .badge-critical { background: #3d1414; color: #f87171; border: 1px solid #5c1f1f; }
    .factory-card {
        background: linear-gradient(145deg, #1a1f2b, #151922);
        border: 1px solid #2a3040; border-radius: 14px; padding: 20px;
        margin-bottom: 12px;
    }
    .rec-high { border-left: 4px solid #f87171; padding: 10px 14px; background: #1a1315; border-radius: 6px; margin-bottom: 8px;}
    .rec-medium { border-left: 4px solid #facc15; padding: 10px 14px; background: #1a1810; border-radius: 6px; margin-bottom: 8px;}
    .rec-low { border-left: 4px solid #4ade80; padding: 10px 14px; background: #101a13; border-radius: 6px; margin-bottom: 8px;}
    h1, h2, h3 { color: #e5e9f0; }
</style>
""", unsafe_allow_html=True)


def badge(status: str) -> str:
    cls = {"Healthy": "badge-healthy", "Warning": "badge-warning", "Critical": "badge-critical",
           "LOW": "badge-healthy", "MEDIUM": "badge-warning", "HIGH": "badge-critical"}.get(status, "badge-warning")
    return f'<span class="badge {cls}">{status}</span>'


# ----------------------------------------------------------------------------
# CACHED LOADERS
# ----------------------------------------------------------------------------
@st.cache_resource
def get_models():
    failure_model, failure_scaler = load_failure_model()
    production_model, production_scaler = load_production_model()
    anomaly_model, anomaly_scaler, anomaly_threshold, anomaly_eval = load_anomaly_model()
    forecast_model, forecast_meta = load_forecast_model()
    return {
        "failure_model": failure_model, "failure_scaler": failure_scaler,
        "production_model": production_model, "production_scaler": production_scaler,
        "anomaly_model": anomaly_model, "anomaly_scaler": anomaly_scaler,
        "anomaly_threshold": anomaly_threshold, "anomaly_eval": anomaly_eval,
        "forecast_model": forecast_model, "forecast_meta": forecast_meta,
    }


@st.cache_data
def get_machine_data():
    return load_machine_data(os.path.join(DATA_DIR, "machine_data.csv"))


@st.cache_data
def get_production_data():
    return load_production_data(os.path.join(DATA_DIR, "production_data.csv"))


@st.cache_data
def get_global_importance(_model):
    return global_feature_importance(_model)


def full_machine_assessment(models, feature_row: dict, target_production: float):
    """Runs ALL models for a given feature row and returns a combined assessment dict."""
    failure_result = predict_failure(models["failure_model"], models["failure_scaler"], feature_row)
    production_result = predict_production(
        models["production_model"], models["production_scaler"], feature_row, target_production
    )

    # Build a synthetic short sequence around the current reading for anomaly scoring
    base = np.array([feature_row[f] for f in ANOMALY_FEATURES])
    seq = np.tile(base, (30, 1)) + np.random.normal(0, 0.15, (30, len(ANOMALY_FEATURES))) * base * 0.02
    anomaly_result = score_sequence(
        models["anomaly_model"], models["anomaly_scaler"], models["anomaly_threshold"], seq, ANOMALY_FEATURES
    )

    top_factors = explain_prediction(models["failure_model"], models["failure_scaler"], feature_row)
    health = compute_health_score(
        failure_result["failure_probability"], anomaly_result["anomaly_score_pct"],
        production_result["efficiency_pct"], feature_row["Tool_Wear"],
    )
    root_cause = root_cause_analysis(feature_row, top_factors)
    recs = generate_recommendations(feature_row, failure_result["failure_probability"], anomaly_result["status"])

    return {
        "failure": failure_result, "production": production_result, "anomaly": anomaly_result,
        "top_factors": top_factors, "health": health, "root_cause": root_cause, "recommendations": recs,
    }


# ----------------------------------------------------------------------------
# SIDEBAR NAVIGATION
# ----------------------------------------------------------------------------
st.sidebar.markdown("## 🏭 FactoryMind AI")
st.sidebar.caption("Predictive Maintenance & Production Optimization")
page = st.sidebar.radio(
    "Navigate",
    ["Dashboard", "Machines", "Predictions", "Anomalies", "Production",
     "Explainability", "Recommendations", "Simulation", "Model Performance"],
    label_visibility="collapsed",
)
st.sidebar.markdown("---")

models = get_models()
machine_df = get_machine_data()
production_df = get_production_data()

latest = machine_df.groupby("Machine_ID").tail(1).reset_index(drop=True)

# Precompute a full assessment for every machine's latest snapshot (used across pages)
@st.cache_data
def compute_all_assessments(_models, latest_df: pd.DataFrame):
    results = {}
    for _, row in latest_df.iterrows():
        feature_row = {f: row[f] for f in FAILURE_FEATURES}
        target = row["Target_Production"]
        results[row["Machine_ID"]] = full_machine_assessment(_models, feature_row, target)
    return results

all_assessments = compute_all_assessments(models, latest)

machine_ids = sorted(latest["Machine_ID"].unique())
selected_machine = st.sidebar.selectbox("Selected Machine", machine_ids)
st.sidebar.markdown("---")
st.sidebar.caption("DETECT → PREDICT → EXPLAIN → RECOMMEND")


# ============================================================================
# PAGE: DASHBOARD (Overview)
# ============================================================================
if page == "Dashboard":
    st.title("Factory Overview")
    st.caption("Real-time AI-driven health and production summary across all machines")

    statuses = [a["health"]["status"] for a in all_assessments.values()]
    healthy_count = statuses.count("Healthy")
    warning_count = statuses.count("Warning")
    critical_count = statuses.count("Critical")
    active_anomalies = sum(1 for a in all_assessments.values() if a["anomaly"]["status"] != "NORMAL")
    avg_health = round(np.mean([a["health"]["health_score"] for a in all_assessments.values()]), 1)
    predicted_total_production = round(sum(a["production"]["predicted_production"] for a in all_assessments.values()))

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Total Machines", len(machine_ids))
    c2.metric("Healthy Machines", healthy_count)
    c3.metric("Machines At Risk", warning_count + critical_count)
    c4.metric("Active Anomalies", active_anomalies)
    c5.metric("Avg Machine Health", f"{avg_health}%")
    c6.metric("Predicted Production", f"{predicted_total_production:,} units")

    st.markdown("### Machine Status Grid")
    cols = st.columns(4)
    for i, mid in enumerate(machine_ids):
        a = all_assessments[mid]
        with cols[i % 4]:
            st.markdown(f"""
            <div class="factory-card">
                <b>{mid}</b> {badge(a['health']['status'])}<br>
                <span style="color:#8b93a7; font-size:0.85rem;">Health: {a['health']['health_score']}%</span><br>
                <span style="color:#8b93a7; font-size:0.85rem;">Failure Risk: {a['failure']['risk_level']}</span>
            </div>
            """, unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("### Health Score Distribution")
        health_scores = pd.DataFrame(
            [{"Machine": mid, "Health": all_assessments[mid]["health"]["health_score"]} for mid in machine_ids]
        )
        fig = px.bar(health_scores, x="Machine", y="Health", color="Health",
                     color_continuous_scale=["#f87171", "#facc15", "#4ade80"], range_color=[0, 100])
        fig.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                           height=350, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown("### Risk Level Breakdown")
        risk_counts = pd.Series([a["failure"]["risk_level"] for a in all_assessments.values()]).value_counts()
        fig2 = go.Figure(data=[go.Pie(
            labels=risk_counts.index, values=risk_counts.values, hole=0.55,
            marker=dict(colors=["#f87171" if l == "HIGH" else "#facc15" if l == "MEDIUM" else "#4ade80" for l in risk_counts.index]),
        )])
        fig2.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                            height=350, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig2, use_container_width=True)


# ============================================================================
# PAGE: MACHINES (monitoring list + detail)
# ============================================================================
elif page == "Machines":
    st.title("Machine Monitoring")

    table_rows = []
    for mid in machine_ids:
        a = all_assessments[mid]
        table_rows.append({
            "Machine": mid, "Status": a["health"]["status"], "Health Score": a["health"]["health_score"],
            "Failure Risk": a["failure"]["risk_level"], "Anomaly Status": a["anomaly"]["status"],
            "Predicted Production": a["production"]["predicted_production"],
        })
    table_df = pd.DataFrame(table_rows)
    st.dataframe(table_df, use_container_width=True, hide_index=True)

    st.markdown(f"### Detail: {selected_machine}")
    row = latest[latest["Machine_ID"] == selected_machine].iloc[0]
    a = all_assessments[selected_machine]

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Sensor Readings**")
        st.write(pd.DataFrame({
            "Sensor": ["Temperature", "Vibration", "Pressure", "Rotational Speed", "Torque", "Tool Wear"],
            "Value": [row["Temperature"], row["Vibration"], row["Pressure"], row["Rotational_Speed"],
                      row["Torque"], row["Tool_Wear"]],
        }).set_index("Sensor"))
    with c2:
        st.markdown("**Operating Context**")
        st.write(pd.DataFrame({
            "Field": ["Operating Hours", "Machine Age (yrs)", "Maintenance Count", "Production Load"],
            "Value": [row["Operating_Hours"], row["Machine_Age_Years"], row["Maintenance_Count"], row["Production_Load"]],
        }).set_index("Field"))
    with c3:
        st.markdown("**AI Assessment**")
        st.metric("Health Score", f"{a['health']['health_score']}%")
        st.markdown(f"Status: {badge(a['health']['status'])}", unsafe_allow_html=True)
        st.markdown(f"Failure Risk: {badge(a['failure']['risk_level'])}", unsafe_allow_html=True)
        st.caption(f"Anomaly: {a['anomaly']['status']} ({a['anomaly']['anomaly_score_pct']}%)")


# ============================================================================
# PAGE: PREDICTIONS
# ============================================================================
elif page == "Predictions":
    st.title("AI Predictions")
    st.caption(f"Machine: {selected_machine}")
    a = all_assessments[selected_machine]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Failure Probability", f"{a['failure']['failure_probability']*100:.1f}%")
    c2.metric("Anomaly Score", f"{a['anomaly']['anomaly_score_pct']}%")
    c3.metric("Predicted Production", f"{a['production']['predicted_production']} units")
    c4.metric("Machine Health", f"{a['health']['health_score']}%")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Failure Prediction")
        st.markdown(f"**Prediction:** {a['failure']['prediction']}")
        st.markdown(f"**Risk Level:** {badge(a['failure']['risk_level'])}", unsafe_allow_html=True)
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=a['failure']['failure_probability'] * 100,
            gauge={"axis": {"range": [0, 100]},
                   "bar": {"color": "#f87171" if a['failure']['risk_level'] == "HIGH" else "#facc15" if a['failure']['risk_level'] == "MEDIUM" else "#4ade80"},
                   "steps": [{"range": [0, 33], "color": "#10381f"}, {"range": [33, 66], "color": "#3d3010"}, {"range": [66, 100], "color": "#3d1414"}]},
            title={"text": "Failure Probability %"},
        ))
        fig.update_layout(template="plotly_dark", height=280, margin=dict(l=20, r=20, t=50, b=10),
                           paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### Production Performance")
        st.markdown(f"**Target Production:** {a['production']['target_production']} units")
        st.markdown(f"**Predicted Production:** {a['production']['predicted_production']} units")
        st.markdown(f"**Efficiency:** {a['production']['efficiency_pct']}%")
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(x=["Target", "Predicted"],
                               y=[a['production']['target_production'], a['production']['predicted_production']],
                               marker_color=["#3b82f6", "#4ade80"]))
        fig2.update_layout(template="plotly_dark", height=280, margin=dict(l=20, r=20, t=20, b=10),
                            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("#### Machine Health Score")
    fig3 = go.Figure(go.Indicator(
        mode="gauge+number", value=a['health']['health_score'],
        gauge={"axis": {"range": [0, 100]},
               "bar": {"color": "#4ade80" if a['health']['status'] == "Healthy" else "#facc15" if a['health']['status'] == "Warning" else "#f87171"}},
        title={"text": f"Health: {a['health']['status']}"},
    ))
    fig3.update_layout(template="plotly_dark", height=260, margin=dict(l=20, r=20, t=50, b=10), paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig3, use_container_width=True)


# ============================================================================
# PAGE: ANOMALIES
# ============================================================================
elif page == "Anomalies":
    st.title("Anomaly Detection")
    st.caption("LSTM Autoencoder — reconstruction-error based anomaly scoring")

    a = all_assessments[selected_machine]
    c1, c2, c3 = st.columns(3)
    c1.metric("Anomaly Score", f"{a['anomaly']['anomaly_score_pct']}%")
    c2.metric("Reconstruction Error", f"{a['anomaly']['reconstruction_error']}")
    c3.metric("Status", a['anomaly']['status'])

    if a['anomaly']['status'] != "NORMAL":
        st.error(f"⚠️ {a['anomaly']['status']} — sensor behavior deviates from learned normal patterns for {selected_machine}")
    else:
        st.success(f"✅ {selected_machine} sensor behavior is within normal learned patterns")

    st.markdown("#### Contributing Sensors")
    contrib_df = pd.DataFrame(
        [{"Sensor": k, "Contribution %": v} for k, v in a['anomaly']['feature_contributions'].items()]
    ).sort_values("Contribution %", ascending=True)
    fig = px.bar(contrib_df, x="Contribution %", y="Sensor", orientation="h", color="Contribution %",
                 color_continuous_scale=["#4ade80", "#facc15", "#f87171"])
    fig.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                       height=300, margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Anomaly Status — All Machines")
    anomaly_rows = [{"Machine": mid, "Anomaly Score %": all_assessments[mid]["anomaly"]["anomaly_score_pct"],
                      "Status": all_assessments[mid]["anomaly"]["status"]} for mid in machine_ids]
    st.dataframe(pd.DataFrame(anomaly_rows), use_container_width=True, hide_index=True)

    st.markdown("#### Reconstruction Error Distribution (training/eval set)")
    eval_data = models["anomaly_eval"]
    err_df = pd.DataFrame({"error": eval_data["all_errors"], "label": eval_data["labels"]})
    err_df["label"] = err_df["label"].map({0: "Normal", 1: "Anomalous (injected)"})
    fig2 = px.histogram(err_df, x="error", color="label", barmode="overlay", nbins=40,
                         color_discrete_map={"Normal": "#4ade80", "Anomalous (injected)": "#f87171"})
    fig2.add_vline(x=models["anomaly_threshold"], line_dash="dash", line_color="#facc15",
                   annotation_text="Threshold")
    fig2.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                        height=320, margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig2, use_container_width=True)


# ============================================================================
# PAGE: PRODUCTION (analytics + forecast)
# ============================================================================
elif page == "Production":
    st.title("Production Analytics")
    st.caption(f"Machine: {selected_machine}")

    machine_prod = production_df[production_df["Machine_ID"] == selected_machine].sort_values("Day")
    a = all_assessments[selected_machine]

    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted Production", f"{a['production']['predicted_production']} units")
    c2.metric("Efficiency", f"{a['production']['efficiency_pct']}%")
    c3.metric("Target Production", f"{a['production']['target_production']} units")

    st.markdown("#### Historical Production")
    fig = px.line(machine_prod, x="Day", y="Production_Output", markers=False)
    fig.update_traces(line_color="#3b82f6")
    fig.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                       height=320, margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Future Production Forecast (GRU)")
    recent_values = machine_prod["Production_Output"].values[-WINDOW:]
    forecast_result = forecast_next(models["forecast_model"], models["forecast_meta"], selected_machine, recent_values)

    hist_days = machine_prod["Day"].values[-WINDOW:]
    future_days = [int(hist_days[-1]) + i + 1 for i in range(len(forecast_result["forecast"]))]

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=hist_days, y=recent_values, mode="lines+markers", name="Historical",
                               line=dict(color="#3b82f6")))
    fig2.add_trace(go.Scatter(x=[hist_days[-1]] + future_days, y=[recent_values[-1]] + forecast_result["forecast"],
                               mode="lines+markers", name="Forecast", line=dict(color="#f97316", dash="dash")))
    fig2.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                        height=320, margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig2, use_container_width=True)

    if forecast_result["trend"] == "DECLINING":
        st.warning("⚠️ WARNING: Declining production trend detected.")
    elif forecast_result["trend"] == "IMPROVING":
        st.success("📈 Production trend improving.")
    else:
        st.info("Production trend stable.")

    st.markdown("#### All Machines — Actual vs Predicted (latest snapshot)")
    compare_rows = [{"Machine": mid, "Actual (last recorded)": latest[latest["Machine_ID"] == mid]["Production_Output"].values[0],
                      "Predicted": all_assessments[mid]["production"]["predicted_production"]} for mid in machine_ids]
    compare_df = pd.DataFrame(compare_rows)
    fig3 = go.Figure()
    fig3.add_trace(go.Bar(x=compare_df["Machine"], y=compare_df["Actual (last recorded)"], name="Actual", marker_color="#3b82f6"))
    fig3.add_trace(go.Bar(x=compare_df["Machine"], y=compare_df["Predicted"], name="Predicted", marker_color="#4ade80"))
    fig3.update_layout(barmode="group", template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                        height=340, margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig3, use_container_width=True)


# ============================================================================
# PAGE: EXPLAINABILITY
# ============================================================================
elif page == "Explainability":
    st.title("Explainable AI")
    st.caption("SHAP-based explanations — model-derived, not rule-based")

    a = all_assessments[selected_machine]
    st.markdown(f"### Why is {selected_machine} at risk?")

    if a["top_factors"]:
        factors_df = pd.DataFrame(a["top_factors"]).rename(
            columns={"feature": "Factor", "contribution_pct": "Contribution %"}
        ).sort_values("Contribution %", ascending=True)
        fig = px.bar(factors_df, x="Contribution %", y="Factor", orientation="h", color="Contribution %",
                     color_continuous_scale=["#4ade80", "#facc15", "#f87171"])
        fig.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                           height=320, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No significant risk-increasing factors detected for this machine.")

    st.markdown("### Global Feature Importance (SHAP, sampled across fleet)")
    importance = get_global_importance(models["failure_model"])
    imp_df = pd.DataFrame(importance).rename(columns={"feature": "Feature", "importance_pct": "Importance %"})
    imp_df = imp_df.sort_values("Importance %", ascending=True)
    fig2 = px.bar(imp_df, x="Importance %", y="Feature", orientation="h", color_discrete_sequence=["#3b82f6"])
    fig2.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                        height=380, margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### Root Cause Analysis")
    row = latest[latest["Machine_ID"] == selected_machine].iloc[0]
    st.markdown("**Detected Issues (rule-based, from sensor thresholds):**")
    for issue in a["root_cause"]["detected_issues"]:
        st.markdown(f"- {issue}")
    st.markdown(f"**Probable Root Cause:** {a['root_cause']['probable_root_cause']}")
    st.markdown(f"**Model-Derived Top Factors:** {', '.join(a['root_cause']['model_derived_top_factors']) or 'None'}")


# ============================================================================
# PAGE: RECOMMENDATIONS
# ============================================================================
elif page == "Recommendations":
    st.title("AI Maintenance Recommendations")
    st.caption(f"Machine: {selected_machine}")

    a = all_assessments[selected_machine]
    for rec in a["recommendations"]:
        css_class = {"HIGH": "rec-high", "MEDIUM": "rec-medium", "LOW": "rec-low"}[rec["severity"]]
        st.markdown(f"""
        <div class="{css_class}">
            {badge(rec['severity'])} &nbsp; <b>{rec['detected_issue']}</b><br>
            <span style="color:#c7ccd6;">Action: {rec['recommended_action']}</span><br>
            <span style="color:#8b93a7; font-size:0.85rem;">Reason: {rec['reason']}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### Fleet-wide Recommendations Summary")
    all_recs = []
    for mid in machine_ids:
        for rec in all_assessments[mid]["recommendations"]:
            all_recs.append({"Machine": mid, **rec})
    recs_df = pd.DataFrame(all_recs)
    recs_df = recs_df[recs_df["severity"] != "LOW"] if (recs_df["severity"] != "LOW").any() else recs_df
    st.dataframe(recs_df.rename(columns={
        "severity": "Severity", "detected_issue": "Detected Issue",
        "recommended_action": "Recommended Action", "reason": "Reason",
    }), use_container_width=True, hide_index=True)


# ============================================================================
# PAGE: SIMULATION
# ============================================================================
elif page == "Simulation":
    st.title("Machine Simulation")
    st.caption("Adjust sensor conditions and watch AI predictions update live")

    row = latest[latest["Machine_ID"] == selected_machine].iloc[0]

    st.markdown("#### Adjust Machine Conditions")
    sc1, sc2 = st.columns(2)
    with sc1:
        sim_temp = st.slider("Temperature (°C)", 20.0, 130.0, float(row["Temperature"]), 0.5)
        sim_vib = st.slider("Vibration (mm/s)", 0.1, 12.0, float(row["Vibration"]), 0.1)
        sim_pressure = st.slider("Pressure (bar)", 5.0, 70.0, float(row["Pressure"]), 0.5)
        sim_speed = st.slider("Rotational Speed (RPM)", 900.0, 1600.0, float(row["Rotational_Speed"]), 5.0)
    with sc2:
        sim_torque = st.slider("Torque (Nm)", 5.0, 90.0, float(row["Torque"]), 0.5)
        sim_wear = st.slider("Tool Wear (%)", 0.0, 100.0, float(row["Tool_Wear"]), 1.0)
        sim_hours = st.slider("Operating Hours", 0.0, 3000.0, float(row["Operating_Hours"]), 10.0)
        sim_load = st.slider("Production Load", 0.2, 1.2, float(row["Production_Load"]), 0.05)

    run = st.button("▶ RUN SIMULATION", type="primary", use_container_width=True)

    sim_feature_row = {
        "Temperature": sim_temp, "Vibration": sim_vib, "Pressure": sim_pressure,
        "Rotational_Speed": sim_speed, "Torque": sim_torque, "Tool_Wear": sim_wear,
        "Operating_Hours": sim_hours, "Machine_Age_Years": float(row["Machine_Age_Years"]),
        "Maintenance_Count": int(row["Maintenance_Count"]), "Production_Load": sim_load,
    }

    if run or "sim_last" in st.session_state:
        result = full_machine_assessment(models, sim_feature_row, float(row["Target_Production"]))
        st.session_state["sim_history"] = st.session_state.get("sim_history", [])
        if run:
            st.session_state["sim_history"].append({
                "Vibration": sim_vib, "Failure Risk %": round(result["failure"]["failure_probability"] * 100, 1),
            })
            st.session_state["sim_history"] = st.session_state["sim_history"][-10:]
        st.session_state["sim_last"] = result

        result = st.session_state["sim_last"]
        st.markdown("---")
        st.markdown("### Live AI Assessment")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Failure Probability", f"{result['failure']['failure_probability']*100:.1f}%")
        c2.metric("Anomaly Score", f"{result['anomaly']['anomaly_score_pct']}%")
        c3.metric("Predicted Production", f"{result['production']['predicted_production']} units")
        c4.metric("Machine Health", f"{result['health']['health_score']}%")

        cc1, cc2 = st.columns(2)
        with cc1:
            st.markdown(f"**Risk Level:** {badge(result['failure']['risk_level'])}", unsafe_allow_html=True)
            st.markdown(f"**Health Status:** {badge(result['health']['status'])}", unsafe_allow_html=True)
            st.markdown(f"**Anomaly Status:** {result['anomaly']['status']}")
        with cc2:
            if result["top_factors"]:
                st.markdown("**Top Risk Factors:**")
                for f in result["top_factors"][:3]:
                    st.markdown(f"- {f['feature']}: {f['contribution_pct']}%")

        st.markdown("### AI Recommendations for Simulated State")
        for rec in result["recommendations"][:3]:
            css_class = {"HIGH": "rec-high", "MEDIUM": "rec-medium", "LOW": "rec-low"}[rec["severity"]]
            st.markdown(f"""
            <div class="{css_class}">
                {badge(rec['severity'])} &nbsp; <b>{rec['detected_issue']}</b><br>
                <span style="color:#c7ccd6;">Action: {rec['recommended_action']}</span>
            </div>
            """, unsafe_allow_html=True)

        if len(st.session_state.get("sim_history", [])) > 1:
            st.markdown("### Simulation History (this session)")
            hist_df = pd.DataFrame(st.session_state["sim_history"])
            fig = px.line(hist_df, y="Failure Risk %", markers=True)
            fig.update_traces(line_color="#f87171")
            fig.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                               height=280, margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Adjust sliders above and click RUN SIMULATION to see live AI predictions.")


# ============================================================================
# PAGE: MODEL PERFORMANCE
# ============================================================================
elif page == "Model Performance":
    st.title("Model Performance")
    st.caption("All metrics computed on held-out test data — nothing fabricated")

    import joblib
    failure_eval = joblib.load(os.path.join(MODEL_DIR, "failure_eval.pkl"))
    production_eval = joblib.load(os.path.join(MODEL_DIR, "production_eval.pkl"))
    anomaly_eval = models["anomaly_eval"]
    forecast_meta = models["forecast_meta"]

    st.markdown("### Failure Classification Model")
    fm = failure_eval["metrics"]
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Accuracy", fm["accuracy"])
    c2.metric("Precision", fm["precision"])
    c3.metric("Recall", fm["recall"])
    c4.metric("F1 Score", fm["f1"])
    c5.metric("ROC-AUC", fm["roc_auc"])
    st.caption(f"Model type: {fm['model_type']}")

    cm = np.array(failure_eval["confusion_matrix"])
    fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                        labels=dict(x="Predicted", y="Actual"),
                        x=["No Failure", "Failure"], y=["No Failure", "Failure"])
    fig_cm.update_layout(template="plotly_dark", height=340, margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig_cm, use_container_width=True)

    fpr, tpr = failure_eval["roc_curve"]
    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="ROC Curve", line=dict(color="#4ade80")))
    fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random", line=dict(dash="dash", color="#8b93a7")))
    fig_roc.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                           height=340, margin=dict(l=10, r=10, t=20, b=10),
                           xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
    st.plotly_chart(fig_roc, use_container_width=True)

    st.markdown("---")
    st.markdown("### Production Regression Model")
    pm = production_eval
    c1, c2, c3 = st.columns(3)
    c1.metric("MAE", pm["mae"])
    c2.metric("RMSE", pm["rmse"])
    c3.metric("R² Score", pm["r2"])
    st.caption(f"Model type: {pm['model_type']}")

    st.markdown("---")
    st.markdown("### Anomaly Detection (LSTM Autoencoder)")
    am = anomaly_eval["metrics"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ROC-AUC", am["roc_auc"])
    c2.metric("Precision", am["precision"])
    c3.metric("Recall", am["recall"])
    c4.metric("F1 Score", am["f1"])
    st.caption(f"Reconstruction error threshold: {am['threshold']} — trained on {am['n_train_normal_sequences']} normal sequences")

    st.markdown("---")
    st.markdown("### Production Forecasting (GRU)")
    st.metric("Normalized MAE (validation)", forecast_meta["mae_normalized"])
    st.caption(f"Window: {forecast_meta['window']} days → Horizon: {forecast_meta['horizon']} days")
