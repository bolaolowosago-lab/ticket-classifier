import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]

st.set_page_config(page_title="IT Ticket Classifier", page_icon="🎫")
st.title("🎫 IT Ticket Classifier")
st.caption("Paste a support ticket and the model predicts its category.")


@st.cache_resource
def load_model():
    return joblib.load(ROOT / "models" / "baseline.joblib")


pipe = load_model()["pipeline"]

examples = {
    "Printer": "The printer on the 2nd floor keeps showing a paper jam error but there is no paper stuck.",
    "Password": "I got locked out of my account after too many login attempts, please reset my password.",
    "Network": "Wifi keeps dropping every few minutes in the conference room.",
}
choice = st.selectbox("Try an example (or write your own below)", ["(custom)"] + list(examples))
text = st.text_area("Ticket text", value=examples.get(choice, ""), height=150)

if st.button("Classify") and text.strip():
    probs = pipe.predict_proba([text])[0]
    df = pd.DataFrame({"category": pipe.classes_, "probability": probs})
    df = df.sort_values("probability", ascending=False).reset_index(drop=True)

    st.subheader(f"Prediction: {df.loc[0, 'category']}")
    st.write(f"Confidence: {df.loc[0, 'probability']:.1%}")
    if df.loc[0, "probability"] < 0.5:
        st.warning("Low confidence. A human should look at this one.")
    st.bar_chart(df.head(5).set_index("category"))

metrics_path = ROOT / "results" / "baseline_metrics.json"
if metrics_path.exists():
    with st.expander("Model performance (held-out test set)"):
        st.json(json.loads(metrics_path.read_text()))
