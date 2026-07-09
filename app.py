"""
Zbulues i Lajmeve te Rreme (Fake News Detector) - Almira Mecaj, Diplome ML
Streamlit demo app per pjesen praktike te punimit.

SI TA PERDORESH:
1. Instalo varesite:  pip install -r requirements.txt
2. Modeli XLM-R (fine-tuned ne shqip, ~95.5% accuracy ne al_test) ngarkohet
   direkt nga Hugging Face Hub: almira123/xlmr-albanian-fake-news
   (s'ka nevoje per download manual, huggingface_hub e ben vete).
3. Run:  streamlit run app.py

Per te shtuar edhe mBERT si opsion te dyte: uploado modelin e mBERT-it fine-tuned
edhe ate ne Hugging Face Hub (njesoj si per XLM-R), pastaj shto nje rresht te ri
te MODELS me poshte me repo_id-ne perkatese.
"""

import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ---------------------------------------------------------------------------
# KONFIGURIMI I MODELEVE - ndrysho path-et kur te kesh checkpoint-et gati
# ---------------------------------------------------------------------------
MODELS = {
    "XLM-R (fine-tuned shqip, ~95.5% accuracy)": {
        "path": "almira123/xlmr-albanian-fake-news",
        "description": "xlm-roberta-base, fine-tuned EN + AL (Hugging Face Hub)",
    },
    # Shto ketu edhe mBERT nese e uploadon ne Hugging Face Hub:
    # "mBERT (fine-tuned shqip, ~91.1% accuracy)": {
    #     "path": "almira123/mbert-albanian-fake-news",
    #     "description": "bert-base-multilingual-cased, fine-tuned EN + AL (Hugging Face Hub)",
    # },
}

LABELS = {0: "REAL", 1: "FAKE"}

# ---------------------------------------------------------------------------
# UI SETUP
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Zbulues i Lajmeve te Rreme", page_icon="📰", layout="centered")

st.title("📰 Zbulues i Lajmeve te Rreme")
st.caption("Prototip per punimin e diplomes — Evaluating Bias and Fairness on Albanian Fake News (Almira Mecaj)")

with st.sidebar:
    st.header("Cilesimet")
    model_choice = st.selectbox("Zgjidh modelin", list(MODELS.keys()))
    st.markdown("---")
    st.markdown(
        "**Si funksionon:** modeli lexon tekstin e artikullit dhe jep "
        "parashikimin Fake/Real, bashke me nje shkalle sigurie (confidence)."
    )
    st.markdown(f"Modeli aktual: `{MODELS[model_choice]['description']}`")


@st.cache_resource(show_spinner="Duke ngarkuar modelin...")
def load_model(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    return tokenizer, model


def predict(text: str, model_path: str):
    tokenizer, model = load_model(model_path)
    inputs = tokenizer(text, truncation=True, padding=True, max_length=256, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=1)[0]
    pred_id = int(torch.argmax(probs))
    return LABELS.get(pred_id, str(pred_id)), float(probs[pred_id]), probs.tolist()


# ---------------------------------------------------------------------------
# INPUT
# ---------------------------------------------------------------------------
text = st.text_area(
    "Ngjit tekstin e nje lajmi ne shqip:",
    height=220,
    placeholder="P.sh. ngjit titullin dhe permbajtjen e nje artikulli lajmesh...",
)

col1, col2 = st.columns([1, 3])
with col1:
    analyze = st.button("🔍 Analizo", type="primary", use_container_width=True)

if analyze:
    if not text.strip():
        st.warning("Fut nje tekst para se te analizosh.")
    else:
        model_path = MODELS[model_choice]["path"]
        try:
            label, confidence, all_probs = predict(text, model_path)

            if label == "FAKE":
                st.error(f"### ⚠️ FAKE — {confidence*100:.1f}% siguri")
            else:
                st.success(f"### ✅ REAL — {confidence*100:.1f}% siguri")

            st.progress(confidence)

            with st.expander("Detaje"):
                st.write(f"**Modeli i perdorur:** {model_choice}")
                st.write(f"**Probabiliteti REAL:** {all_probs[0]*100:.2f}%")
                st.write(f"**Probabiliteti FAKE:** {all_probs[1]*100:.2f}%")

        except OSError:
            st.error(
                f"S'u gjet modeli te path-i `{model_path}`. "
                "Kontrollo variablën MODELS ne krye te app.py dhe vendos path-in "
                "e sakte te checkpoint-it tend (lokal ose HuggingFace Hub)."
            )

st.markdown("---")
st.caption(
    "Kujdes: ky eshte nje prototip akademik per demonstrim, jo nje mjet i verifikuar per "
    "perdorim ne prodhim. Rezultatet duhen interpretuar me kujdes."
)
