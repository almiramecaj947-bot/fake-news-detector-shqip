"""
Zbulues i Lajmeve te Rreme (Fake News Detector) - Almira Mecaj, Diplome ML
Streamlit demo app per pjesen praktike te punimit.
"""

import streamlit as st
import torch
import trafilatura
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODELS = {
    "XLM-R (fine-tuned shqip, ~95.5% accuracy)": {
        "path": "almira123/xlmr-albanian-fake-news",
        "description": "xlm-roberta-base, fine-tuned EN + AL (Hugging Face Hub)",
    },
}

LABELS = {0: "REAL", 1: "FAKE"}

st.set_page_config(page_title="Zbulues i Lajmeve te Rreme", page_icon="📰", layout="centered")
st.title("Zbulues i Lajmeve te Rreme")
st.caption("Prototip per punimin e diplomes (Almira Mecaj)")

with st.sidebar:
    st.header("Cilesimet")
    model_choice = st.selectbox("Zgjidh modelin", list(MODELS.keys()))
    st.markdown("---")
    st.markdown(f"Modeli aktual: `{MODELS[model_choice]['description']}`")


@st.cache_resource(show_spinner="Duke ngarkuar modelin...")
def load_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    return tokenizer, model


def predict(text, model_path):
    tokenizer, model = load_model(model_path)
    inputs = tokenizer(text, truncation=True, padding=True, max_length=256, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=1)[0]
    pred_id = int(torch.argmax(probs))
    return LABELS.get(pred_id, str(pred_id)), float(probs[pred_id]), probs.tolist()


def fetch_article_text(url):
    downloaded = trafilatura.fetch_url(url)
    if downloaded is None:
        raise ValueError("S'u arrit te hapet ky link.")
    extracted = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
    if not extracted or not extracted.strip():
        raise ValueError("S'u gjet tekst artikulli ne kete faqe.")
    return extracted.strip()


input_mode = st.radio("Si do ta japesh lajmin?", ["Ngjit tekstin", "Vendos link (URL)"], horizontal=True)

text = ""

if input_mode == "Ngjit tekstin":
    text = st.text_area("Ngjit tekstin e nje lajmi ne shqip:", height=220)
else:
    url = st.text_input("Vendos linkun e artikullit:", placeholder="https://...")
    if st.button("Merr artikullin nga linku"):
        if not url.strip():
            st.warning("Fut nje link para se te vazhdosh.")
        else:
            try:
                with st.spinner("Duke shkarkuar..."):
                    fetched = fetch_article_text(url.strip())
                st.session_state["fetched_text"] = fetched
                st.success(f"U morren {len(fetched)} karaktere.")
            except ValueError as e:
                st.error(str(e))
    text = st.text_area("Teksti i marre (mund ta redaktosh):", value=st.session_state.get("fetched_text", ""), height=220)

analyze = st.button("Analizo", type="primary")

if analyze:
    if not text.strip():
        st.warning("Fut nje tekst para se te analizosh.")
    else:
        model_path = MODELS[model_choice]["path"]
        try:
            label, confidence, all_probs = predict(text, model_path)
            if label == "FAKE":
                st.error(f"FAKE - {confidence*100:.1f}% siguri")
            else:
                st.success(f"REAL - {confidence*100:.1f}% siguri")
            st.progress(confidence)
            with st.expander("Detaje"):
                st.write(f"Modeli: {model_choice}")
                st.write(f"Probabiliteti REAL: {all_probs[0]*100:.2f}%")
                st.write(f"Probabiliteti FAKE: {all_probs[1]*100:.2f}%")
        except OSError:
            st.error(f"S'u gjet modeli te {model_path}.")

st.markdown("---")
st.caption("Prototip akademik per demonstrim.")
