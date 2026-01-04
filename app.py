import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from datetime import date, timedelta

st.set_page_config(page_title="Mood & energy, day by day", layout="wide")

# ----------------------------
# Friendly labels
# ----------------------------
PHASES = ["menstruation", "late-follicular", "ovulation", "luteal"]

MOOD_LABELS = {0: "Not great", 1: "Okay", 2: "Good"}
ENERGY_LABELS = {0: "Low", 1: "Steady", 2: "High"}

PHASE_FRIENDLY = {
    "menstruation": "Menstruation",
    "late-follicular": "Late follicular",
    "ovulation": "Ovulation",
    "luteal": "Luteal",
}


def label_series(s: pd.Series, mapping: dict) -> pd.Series:
    return s.map(mapping)


# ----------------------------
# Mock data + mock model
# ----------------------------
def make_mock_user_data(start_date: date, days: int = 90, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = [start_date + timedelta(days=i) for i in range(days)]

    # simple repeating cycle phase pattern (placeholder)
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
    # mood: 0,1,2
    mood_latent = (
        0.25 * (sleep_duration - 7) -
        0.018 * (stress_score - 45) +
        0.00003 * (steps - 8500) +
        rng.normal(0, 0.35, days)
    )
    phase_effect = {"menstruation": -0.20, "late-follicular": 0.05, "ovulation": 0.12, "luteal": -0.08}
    mood_latent += np.array([phase_effect[p] for p in phases])
    mood_obs = np.digitize(mood_latent, [-0.2, 0.35])  # -> {0,1,2}

    # energy physical/mental: 0,1,2
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
    Demo-only predictor. Replace with your friend's model later.
    """
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

    explanations = {
        "Sleep (hours)": float(0.22 * (row["sleep_duration"] - 7)),
        "Stress level": float(-0.02 * ((row["stress_score"] - 45) / 10)),
        "Activity (steps)": float(0.00003 * (row["activity_steps"] - 8500)),
        f"Cycle phase: {PHASE_FRIENDLY.get(row['cycle_phase'], row['cycle_phase'])}": float(phase_bonus.get(row["cycle_phase"], 0.0)),
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


# ----------------------------
# Sidebar controls (friendly wording)
# ----------------------------
st.sidebar.title("Your wellbeing view")
st.sidebar.caption("Explore patterns across your cycle. This is not medical advice.")

user_seed = st.sidebar.number_input("Demo participant", min_value=1, max_value=42, value=7, step=1)
start = st.sidebar.date_input("Start date", value=date(2025, 10, 1))
days = st.sidebar.slider("Days shown", 30, 120, 90, 5)

target_view = st.sidebar.selectbox("Color the calendar by", ["Mood", "Physical energy", "Mental energy"])
show_observed = st.sidebar.checkbox("Show your check-ins (if available)", value=True)

df = make_mock_user_data(start, days=days, seed=int(user_seed))
df = add_predictions(df)

# ----------------------------
# Main header
# ----------------------------
st.title("Mood & energy, day by day")
st.write("See how sleep, stress, activity, and cycle phase relate to mood and energy. Use the sliders to explore what might change.")
st.caption("Predictions are for exploration only — not diagnosis or medical advice.")
st.info("Right now, predictions are demo-only (mock). Later, we’ll connect your friend’s ML model without changing the UI.")

# ----------------------------
# Quick overview cards (not calling it 'accuracy' to avoid over-claiming)
# ----------------------------
colA, colB, colC, colD = st.columns(4)

agree_mood = (df["mood_pred"] == df["mood_obs"]).mean()
agree_phys = (df["energy_phys_pred"] == df["energy_phys_obs"]).mean()
agree_ment = (df["energy_ment_pred"] == df["energy_ment_obs"]).mean()

colA.metric("Days shown", len(df))
colB.metric("Mood match (demo)", f"{agree_mood:.0%}")
colC.metric("Physical energy match (demo)", f"{agree_phys:.0%}")
colD.metric("Mental energy match (demo)", f"{agree_ment:.0%}")

# ----------------------------
# Layout: Calendar + Timeline + Try changing your day
# ----------------------------
left, right = st.columns([1.15, 0.85])

with left:
    st.subheader("Calendar view")
    st.caption("Each dot is a day. Color shows the predicted level for that day.")

    metric_map = {
        "Mood": ("mood_pred", "mood_obs", MOOD_LABELS),
        "Physical energy": ("energy_phys_pred", "energy_phys_obs", ENERGY_LABELS),
        "Mental energy": ("energy_ment_pred", "energy_ment_obs", ENERGY_LABELS),
    }
    pred_col, obs_col, label_map = metric_map[target_view]

    cal = df.copy()
    cal["day"] = cal["date"].dt.day
    cal["weekday"] = cal["date"].dt.weekday  # Mon=0
    cal["week"] = cal["date"].dt.isocalendar().week.astype(int)

    cal["value"] = cal[pred_col]
    cal["value_label"] = label_series(cal["value"], label_map)
    cal["phase_friendly"] = cal["cycle_phase"].map(PHASE_FRIENDLY)

    if show_observed:
        cal["obs_label"] = label_series(cal[obs_col], label_map)
        cal["hover"] = (
            cal["date"].dt.date.astype(str)
            + "<br><b>Cycle phase:</b> " + cal["phase_friendly"]
            + "<br><b>Predicted:</b> " + cal["value_label"]
            + "<br><b>Your check-in:</b> " + cal["obs_label"]
            + "<br><b>Sleep:</b> " + cal["sleep_duration"].astype(str) + "h"
            + "<br><b>Stress:</b> " + cal["stress_score"].astype(str)
        )
    else:
        cal["hover"] = (
            cal["date"].dt.date.astype(str)
            + "<br><b>Cycle phase:</b> " + cal["phase_friendly"]
            + "<br><b>Predicted:</b> " + cal["value_label"]
            + "<br><b>Sleep:</b> " + cal["sleep_duration"].astype(str) + "h"
            + "<br><b>Stress:</b> " + cal["stress_score"].astype(str)
        )

    fig_cal = px.scatter(
        cal,
        x="weekday",
        y="week",
        color="value",
        hover_name="day",
        hover_data={"hover": True, "weekday": False, "week": False, "value": False},
        title=f"{target_view} (predicted)",
    )
    fig_cal.update_traces(marker=dict(size=18), hovertemplate="%{customdata[0]}<extra></extra>")
    fig_cal.update_layout(
        xaxis=dict(
            tickmode="array",
            tickvals=list(range(7)),
            ticktext=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
            title=""
        ),
        yaxis=dict(autorange="reversed", title="ISO week"),
        legend_title_text="Level (0–2)",
        height=420
    )
    st.plotly_chart(fig_cal, use_container_width=True)

    st.subheader("Trends over time")
    st.caption("Sleep, stress, and activity overlaid with predicted mood (dots).")

    df_line = df.copy()
    df_line["mood_pred_label"] = label_series(df_line["mood_pred"], MOOD_LABELS)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_line["date"], y=df_line["sleep_duration"], name="Sleep (hours)", mode="lines"))
    fig.add_trace(go.Scatter(x=df_line["date"], y=df_line["stress_score"], name="Stress level", mode="lines"))
    fig.add_trace(go.Scatter(x=df_line["date"], y=df_line["activity_steps"], name="Steps (activity)", mode="lines"))

    # mood dots (scaled just to sit in same plot without multi-axis complexity)
    fig.add_trace(go.Scatter(
        x=df_line["date"],
        y=df_line["mood_pred"] * 3000 + 3000,
        name="Predicted mood (dots)",
        mode="markers",
        text=df_line["mood_pred_label"],
        hovertemplate="Predicted mood: %{text}<br>%{x|%Y-%m-%d}<extra></extra>"
    ))

    fig.update_layout(
        title="Signals over time",
        xaxis_title="Date",
        yaxis_title="Signals (mixed units)",
        height=450
    )
    st.plotly_chart(fig, use_container_width=True)


with right:
    st.subheader("Try changing your day")
    st.caption("Adjust sleep, stress, or activity to see how the prediction changes (exploration only).")

    day_choice = st.selectbox(
        "Choose a day",
        options=df["date"].dt.date.tolist(),
        index=min(10, len(df) - 1)
    )

    row = df.loc[df["date"].dt.date == day_choice].iloc[0].copy()

    st.markdown("**Move the sliders below.** The results update instantly.")

    cycle_phase = st.selectbox(
        "Where you are in your cycle",
        PHASES,
        index=PHASES.index(row["cycle_phase"]),
        format_func=lambda p: PHASE_FRIENDLY.get(p, p)
    )
    sleep_duration = st.slider("Hours of sleep last night", 3.0, 10.5, float(row["sleep_duration"]), 0.1)
    sleep_score = st.slider("Sleep quality score", 0, 100, int(row["sleep_score"]), 1)
    stress_score = st.slider("Stress level", 0, 100, int(row["stress_score"]), 1)
    resting_hr = st.slider("Resting heart rate", 40, 95, int(row["resting_hr"]), 1)
    activity_steps = st.slider("Steps (activity)", 0, 25000, int(row["activity_steps"]), 100)
    temp_delta = st.slider("Temperature change vs baseline", -0.5, 0.8, float(row["temp_delta"]), 0.01)

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
    c1.metric("Predicted mood", MOOD_LABELS[pred["pred_mood"]])
    c2.metric("Predicted physical energy", ENERGY_LABELS[pred["pred_energy_physical"]])
    c3.metric("Predicted mental energy", ENERGY_LABELS[pred["pred_energy_mental"]])

    st.markdown("**What seems to be influencing this (rough explanation):**")
    st.caption("This is a simple placeholder explanation. Later you can swap in SHAP or model-based explanations.")

    exp = pd.DataFrame(
        sorted(pred["explanations"].items(), key=lambda x: abs(x[1]), reverse=True),
        columns=["Factor", "Effect (demo)"]
    )
    st.dataframe(exp, use_container_width=True)

    st.divider()
    st.subheader("Details (for research / debugging)")
    st.dataframe(df[[
        "date", "cycle_phase", "sleep_duration", "sleep_score", "stress_score",
        "resting_hr", "activity_steps", "temp_delta",
        "mood_obs", "mood_pred",
        "energy_phys_obs", "energy_phys_pred",
        "energy_ment_obs", "energy_ment_pred"
    ]], use_container_width=True)
