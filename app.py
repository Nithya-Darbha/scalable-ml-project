import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from datetime import date, timedelta

st.set_page_config(page_title="Cycle & Wellbeing Explorer", layout="wide")

# ----------------------------
# Mock data + mock model
# ----------------------------
PHASES = ["menstruation", "late-follicular", "ovulation", "luteal"]

def make_mock_user_data(start_date: date, days: int = 90, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = [start_date + timedelta(days=i) for i in range(days)]

    # simple repeating cycle phase pattern (not medically precise — placeholder)
    phase_cycle = (["menstruation"] * 5 +
                   ["late-follicular"] * 7 +
                   ["ovulation"] * 3 +
                   ["luteal"] * 13)
    phases = [phase_cycle[i % len(phase_cycle)] for i in range(days)]

    sleep_duration = np.clip(rng.normal(7.2, 0.9, days), 4.0, 10.0)
    sleep_score = np.clip(60 + (sleep_duration - 7) * 8 + rng.normal(0, 10, days), 0, 100)
    stress_score = np.clip(rng.normal(45, 15, days), 0, 100)
    resting_hr = np.clip(rng.normal(63, 5, days), 45, 90)
    steps = np.clip(rng.normal(8500, 2500, days), 0, 20000)
    temp_delta = np.clip(rng.normal(0.15, 0.12, days), -0.3, 0.6)

    # "observed" self-reports (mock)
    # mood: 0=bad,1=neutral,2=good
    mood_latent = (
        0.25 * (sleep_duration - 7) -
        0.018 * (stress_score - 45) +
        0.00003 * (steps - 8500) +
        rng.normal(0, 0.35, days)
    )
    phase_effect = {"menstruation": -0.20, "late-follicular": 0.05, "ovulation": 0.12, "luteal": -0.08}
    mood_latent += np.array([phase_effect[p] for p in phases])
    mood_obs = np.digitize(mood_latent, [-0.2, 0.35])  # -> {0,1,2}

    # energy physical/mental: 0=low,1=normal,2=high
    phys_latent = 0.22*(sleep_duration-7) - 0.015*(stress_score-45) + 0.00004*(steps-8500) + rng.normal(0,0.35,days)
    ment_latent = 0.25*(sleep_score-60)/20 - 0.018*(stress_score-45) + rng.normal(0,0.35,days)
    phys_obs = np.digitize(phys_latent, [-0.15, 0.35])
    ment_obs = np.digitize(ment_latent, [-0.15, 0.35])

    df = pd.DataFrame({
        "date": pd.to_datetime(dates),
        "cycle_phase": phases,
        "sleep_duration": sleep_duration.round(2),
        "sleep_score": sleep_score.round(0),
        "stress_score": stress_score.round(0),
        "resting_hr": resting_hr.round(0),
        "activity_steps": steps.round(0),
        "temp_delta": temp_delta.round(2),
        "mood_obs": mood_obs,
        "energy_phys_obs": phys_obs,
        "energy_ment_obs": ment_obs,
    })

    return df


def mock_predict_row(row: pd.Series) -> dict:
    """
    Placeholder: replace this with your friend's model call.
    """
    # simple rule-ish predictor (keeps UI functional)
    phase_bonus = {"menstruation": -0.08, "late-follicular": 0.05, "ovulation": 0.10, "luteal": -0.04}
    z = (
        0.22 * (row["sleep_duration"] - 7) -
        0.02 * ((row["stress_score"] - 45) / 10) +
        0.00003 * (row["activity_steps"] - 8500) +
        phase_bonus.get(row["cycle_phase"], 0.0)
    )

    mood = int(np.clip(np.digitize(z, [-0.15, 0.35]), 0, 2))
    phys = int(np.clip(np.digitize(z + 0.05*(row["sleep_score"]-70)/30, [-0.15, 0.35]), 0, 2))
    ment = int(np.clip(np.digitize(z - 0.03*(row["resting_hr"]-62)/10, [-0.15, 0.35]), 0, 2))

    # lightweight "explanations" proxy for the what-if panel
    explanations = {
        "sleep_duration": float(0.22 * (row["sleep_duration"] - 7)),
        "stress_score": float(-0.02 * ((row["stress_score"] - 45) / 10)),
        "activity_steps": float(0.00003 * (row["activity_steps"] - 8500)),
        f"cycle_phase_{row['cycle_phase']}": float(phase_bonus.get(row["cycle_phase"], 0.0)),
    }

    return {"pred_mood": mood, "pred_energy_physical": phys, "pred_energy_mental": ment, "explanations": explanations}


def add_predictions(df: pd.DataFrame) -> pd.DataFrame:
    preds = df.apply(mock_predict_row, axis=1)
    df = df.copy()
    df["mood_pred"] = [p["pred_mood"] for p in preds]
    df["energy_phys_pred"] = [p["pred_energy_physical"] for p in preds]
    df["energy_ment_pred"] = [p["pred_energy_mental"] for p in preds]
    df["explanations"] = [p["explanations"] for p in preds]
    return df


LABEL_3 = {0: "Low/Bad", 1: "Mid/Neutral", 2: "High/Good"}

def label_series(s: pd.Series) -> pd.Series:
    return s.map(LABEL_3)


# ----------------------------
# Sidebar controls
# ----------------------------
st.sidebar.title("Cycle & Wellbeing Explorer")

user_seed = st.sidebar.number_input("Participant (mock)", min_value=1, max_value=42, value=7, step=1)
start = st.sidebar.date_input("Start date", value=date(2025, 10, 1))
days = st.sidebar.slider("Days shown", 30, 120, 90, 5)

target_view = st.sidebar.selectbox("Calendar coloring", ["Mood", "Physical Energy", "Mental Energy"])
show_observed = st.sidebar.checkbox("Show observed (if available)", value=True)

df = make_mock_user_data(start, days=days, seed=int(user_seed))
df = add_predictions(df)

# ----------------------------
# Header KPIs
# ----------------------------
st.title("Daily Mood & Energy — Cycle-aware Wellbeing Dashboard")
colA, colB, colC, colD = st.columns(4)

acc_mood = (df["mood_pred"] == df["mood_obs"]).mean()
acc_phys = (df["energy_phys_pred"] == df["energy_phys_obs"]).mean()
acc_ment = (df["energy_ment_pred"] == df["energy_ment_obs"]).mean()

colA.metric("Days", len(df))
colB.metric("Mock mood agreement", f"{acc_mood:.0%}")
colC.metric("Mock physical energy agreement", f"{acc_phys:.0%}")
colD.metric("Mock mental energy agreement", f"{acc_ment:.0%}")

st.caption("Note: predictions are currently mocked. Replace `mock_predict_row()` with the real model call later.")

# ----------------------------
# Layout: Calendar + Timeline + What-if
# ----------------------------
left, right = st.columns([1.15, 0.85])

with left:
    st.subheader("Calendar / cycle view")

    metric_map = {
        "Mood": ("mood_pred", "mood_obs"),
        "Physical Energy": ("energy_phys_pred", "energy_phys_obs"),
        "Mental Energy": ("energy_ment_pred", "energy_ment_obs"),
    }
    pred_col, obs_col = metric_map[target_view]

    cal = df.copy()
    cal["day"] = cal["date"].dt.day
    cal["weekday"] = cal["date"].dt.weekday  # Mon=0
    cal["week"] = cal["date"].dt.isocalendar().week.astype(int)
    cal["month"] = cal["date"].dt.to_period("M").astype(str)

    cal["value"] = cal[pred_col]
    cal["value_label"] = label_series(cal["value"])
    if show_observed:
        cal["obs_label"] = label_series(cal[obs_col])
        cal["hover"] = (
            cal["date"].dt.date.astype(str)
            + "<br>Phase: " + cal["cycle_phase"]
            + "<br>Pred: " + cal["value_label"]
            + "<br>Obs: " + cal["obs_label"]
            + "<br>Sleep: " + cal["sleep_duration"].astype(str) + "h"
            + "<br>Stress: " + cal["stress_score"].astype(str)
        )
    else:
        cal["hover"] = (
            cal["date"].dt.date.astype(str)
            + "<br>Phase: " + cal["cycle_phase"]
            + "<br>Pred: " + cal["value_label"]
            + "<br>Sleep: " + cal["sleep_duration"].astype(str) + "h"
            + "<br>Stress: " + cal["stress_score"].astype(str)
        )

    fig_cal = px.scatter(
        cal,
        x="weekday",
        y="week",
        color="value",
        hover_name="day",
        hover_data={"hover": True, "weekday": False, "week": False, "value": False},
        title=f"{target_view} (Predicted)",
    )
    fig_cal.update_traces(marker=dict(size=18), hovertemplate="%{customdata[0]}<extra></extra>")
    fig_cal.update_layout(
        xaxis=dict(
            tickmode="array",
            tickvals=list(range(7)),
            ticktext=["Mon","Tue","Wed","Thu","Fri","Sat","Sun"],
            title=""
        ),
        yaxis=dict(autorange="reversed", title="ISO week"),
        legend_title_text="Level (0-2)"
    )
    st.plotly_chart(fig_cal, use_container_width=True)

    st.subheader("Timeline: signals + predictions")

    df_line = df.copy()
    df_line["mood_pred_label"] = label_series(df_line["mood_pred"])
    df_line["phase"] = df_line["cycle_phase"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_line["date"], y=df_line["sleep_duration"], name="Sleep (h)", mode="lines"))
    fig.add_trace(go.Scatter(x=df_line["date"], y=df_line["stress_score"], name="Stress", mode="lines"))
    fig.add_trace(go.Scatter(x=df_line["date"], y=df_line["activity_steps"], name="Steps", mode="lines"))

    # add predicted mood as markers on a separate axis-ish scale (just scaled for visibility)
    fig.add_trace(go.Scatter(
        x=df_line["date"],
        y=df_line["mood_pred"] * 3000 + 3000,  # scale to sit in plot; you can do multi-axis later
        name="Mood pred (scaled)",
        mode="markers",
        text=df_line["mood_pred_label"],
        hovertemplate="Mood pred: %{text}<br>%{x|%Y-%m-%d}<extra></extra>"
    ))

    fig.update_layout(
        title="Sleep / Stress / Steps with Mood Predictions",
        xaxis_title="Date",
        yaxis_title="Signals (mixed units)",
        height=450
    )
    st.plotly_chart(fig, use_container_width=True)


with right:
    st.subheader("What-if panel (single day)")

    day_choice = st.selectbox(
        "Pick a day",
        options=df["date"].dt.date.tolist(),
        index=min(10, len(df)-1)
    )

    row = df.loc[df["date"].dt.date == day_choice].iloc[0].copy()

    st.markdown("Adjust inputs and see the predicted outcome update immediately:")

    cycle_phase = st.selectbox("Cycle phase", PHASES, index=PHASES.index(row["cycle_phase"]))
    sleep_duration = st.slider("Sleep duration (hours)", 3.0, 10.5, float(row["sleep_duration"]), 0.1)
    sleep_score = st.slider("Sleep score", 0, 100, int(row["sleep_score"]), 1)
    stress_score = st.slider("Stress score", 0, 100, int(row["stress_score"]), 1)
    resting_hr = st.slider("Resting HR", 40, 95, int(row["resting_hr"]), 1)
    activity_steps = st.slider("Steps", 0, 25000, int(row["activity_steps"]), 100)
    temp_delta = st.slider("Temp delta", -0.5, 0.8, float(row["temp_delta"]), 0.01)

    whatif = pd.Series({
        "cycle_phase": cycle_phase,
        "sleep_duration": sleep_duration,
        "sleep_score": sleep_score,
        "stress_score": stress_score,
        "resting_hr": resting_hr,
        "activity_steps": activity_steps,
        "temp_delta": temp_delta
    })

    pred = mock_predict_row(whatif)

    c1, c2, c3 = st.columns(3)
    c1.metric("Mood (pred)", LABEL_3[pred["pred_mood"]])
    c2.metric("Physical energy (pred)", LABEL_3[pred["pred_energy_physical"]])
    c3.metric("Mental energy (pred)", LABEL_3[pred["pred_energy_mental"]])

    st.markdown("**Simple interpretability (placeholder contributions):**")
    exp = pd.DataFrame(
        sorted(pred["explanations"].items(), key=lambda x: abs(x[1]), reverse=True),
        columns=["feature", "contribution"]
    )
    st.dataframe(exp, use_container_width=True)

    st.divider()
    st.subheader("Data table (debug / stakeholder view)")
    st.dataframe(df[[
        "date", "cycle_phase", "sleep_duration", "sleep_score", "stress_score",
        "resting_hr", "activity_steps", "temp_delta",
        "mood_obs", "mood_pred",
        "energy_phys_obs", "energy_phys_pred",
        "energy_ment_obs", "energy_ment_pred"
    ]], use_container_width=True)
