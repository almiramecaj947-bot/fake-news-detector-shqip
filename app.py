"""
Zbulues i Lajmeve te Rreme (Fake News Detector) - Almira Mecaj, Diplome ML
Streamlit demo app per pjesen praktike te punimit:
"Evaluating Bias and Fairness of Multilingual Models on Albanian Fake News"
"""

import streamlit as st
import streamlit.components.v1 as components
import torch
import trafilatura
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def inject_pwa_head():
    """Shton manifest.json dhe ikonen ne <head> te faqes, qe app-i te jete
    'i instalueshem' (Add to Home Screen) ne telefon, si nje app real."""
    components.html(
        """
        <script>
        (function() {
            const parentDoc = window.parent.document;
            if (!parentDoc.querySelector('link[rel="manifest"]')) {
                const link = parentDoc.createElement('link');
                link.rel = 'manifest';
                link.href = './app/static/manifest.json';
                parentDoc.head.appendChild(link);
            }
            if (!parentDoc.querySelector('link[rel="apple-touch-icon"]')) {
                const icon = parentDoc.createElement('link');
                icon.rel = 'apple-touch-icon';
                icon.href = './app/static/icon-192.png';
                parentDoc.head.appendChild(icon);
            }
            if (!parentDoc.querySelector('meta[name="theme-color"]')) {
                const meta = parentDoc.createElement('meta');
                meta.name = 'theme-color';
                meta.content = '#4f46e5';
                parentDoc.head.appendChild(meta);
            }
        })();
        </script>
        """,
        height=0,
        width=0,
    )


# ---------------------------------------------------------------------------
# KONFIGURIMI I MODELEVE
# ---------------------------------------------------------------------------
MODELS = {
    "XLM-R (few-shot, 95.5% accuracy)": {
        "path": "almira123/xlmr-albanian-fake-news",
        "description": "xlm-roberta-base, fine-tuned EN + AL",
    },
    "mBERT (few-shot, 91.1% accuracy)": {
        "path": "almira123/mbert-albanian-fake-news",
        "description": "bert-base-multilingual-cased, fine-tuned EN + AL",
    },
}

LABELS = {0: "REAL", 1: "FAKE"}

EXAMPLES = {
    "📗 Shembull — lajm real": (
        "Adelina e tepron me fustanin e shkurtër Adelina Tahiri është një ndër femrat më "
        "provokuese në mediat rozë. Duke mos hezituar që të pozojë në forma të ndryshme, "
        "këngëtarja duket gjithmonë e më e zjarrtë me stilin e veçantë që ka. Sidomos, në "
        "imazhin e fundit me një fustan të shkurtër dhe të ngushtë ajo e teproi me pozën që "
        "ka realizuar. Theksojmë, ajo kohëve të fundit mungon në projekte muzikore."
    ),
    "📕 Shembull — lajm i rremë": (
        "Kjo është mundësia e ardhjes së mërgimtarëve nga Gjermania. Gazetari i Deutsche "
        "Welle, Bahri Cani, ka folur për mundësitë që kanë kosovarët të cilët jetojnë në "
        "Gjermani për të ardhur drejtë Kosovës për pushime verore. “500 mijë shqiptarë sa "
        "jetojnë në Gjermani dëshirojnë që pushimet e tyre t’i kalojnë në Kosovë, Shqipëri "
        "dhe vendet tjera” — shiko pamjet se si mund të udhëtojnë mërgimtarët për në vendlindje."
    ),
}

# ---------------------------------------------------------------------------
# UI SETUP
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Zbulues i Lajmeve te Rreme", page_icon="📰", layout="centered")
inject_pwa_head()

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }
    .stApp { background: #f4f5fb; }

    #MainMenu, footer, header { visibility: hidden; }

    .main-header {
        padding: 2.2rem 2rem; border-radius: 20px; margin-bottom: 1.4rem;
        background: linear-gradient(135deg, #7c3aed 0%, #4f46e5 55%, #4338ca 100%);
        color: white; box-shadow: 0 12px 30px rgba(79,70,229,0.30);
    }
    .main-header h1 { margin: 0; font-size: 2rem; font-weight: 800; letter-spacing: -0.5px; }
    .main-header p { margin: 0.5rem 0 0.9rem 0; opacity: 0.92; font-size: 1rem; max-width: 620px; line-height: 1.4;}
    .badge {
        display: inline-block; margin-right: 0.4rem; padding: 0.32rem 0.85rem; border-radius: 999px;
        background: rgba(255,255,255,0.18); font-size: 0.78rem; font-weight: 600; backdrop-filter: blur(4px);
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 6px; background: white; padding: 6px; border-radius: 14px;
        box-shadow: 0 2px 10px rgba(17,24,39,0.06);
    }
    .stTabs [data-baseweb="tab"] { border-radius: 10px; padding: 10px 20px; font-weight: 600; color: #4b5563; }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #7c3aed, #4f46e5) !important; color: white !important;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] > div {
        border-radius: 16px;
    }
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: white; border-radius: 16px !important; border: 1px solid #ececf8 !important;
        box-shadow: 0 4px 18px rgba(17,24,39,0.05);
    }

    .section-title {
        font-weight: 700; font-size: 1.05rem; color: #312e81; margin-bottom: 0.4rem;
    }

    .stButton>button {
        border-radius: 10px; font-weight: 600; border: 1px solid #e5e7eb; transition: all 0.15s ease;
    }
    .stButton>button[kind="primary"] {
        background: linear-gradient(135deg, #7c3aed, #4f46e5); border: none;
        box-shadow: 0 4px 14px rgba(79,70,229,0.35);
    }
    .stButton>button:hover { transform: translateY(-1px); }

    textarea, .stTextInput>div>div>input { border-radius: 12px !important; }

    section[data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid #ececf8; }

    .result-card {
        border-radius: 16px; padding: 1.3rem 1.5rem; margin-top: 0.6rem;
        box-shadow: 0 6px 20px rgba(17,24,39,0.08);
    }
    .result-fake { background: linear-gradient(135deg, #fef2f2, #fee2e2); border: 1px solid #fecaca; }
    .result-real { background: linear-gradient(135deg, #f0fdf4, #dcfce7); border: 1px solid #bbf7d0; }
    .result-title-fake { color: #b91c1c; font-size: 1.35rem; font-weight: 800; margin: 0; }
    .result-title-real { color: #15803d; font-size: 1.35rem; font-weight: 800; margin: 0; }
    .result-sub { color: #4b5563; font-size: 0.9rem; margin-top: 0.2rem; }
    </style>

    <div class="main-header">
        <span class="badge">🎓 Punim Diplome</span>
        <span class="badge">🤖 mBERT · XLM-R</span>
        <h1>📰 Zbulues i Lajmeve te Rreme</h1>
        <p>Prototip praktik për punimin "Evaluating Bias and Fairness of Multilingual Models on
        Albanian Fake News" (Almira Mecaj) — analizon artikuj lajmesh në shqip dhe vlerëson
        gjasat që të jenë lajm i rremë.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_analyze, tab_findings = st.tabs(["🔍  Analizo Artikull", "📊  Gjetjet e Kërkimit"])

# ---------------------------------------------------------------------------
# MODEL LOADING / PREDIKIMI
# ---------------------------------------------------------------------------
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


def fetch_article_text(url: str) -> str:
    downloaded = trafilatura.fetch_url(url)
    if downloaded is None:
        raise ValueError("S'u arrit të hapej ky link.")
    extracted = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
    if not extracted or not extracted.strip():
        raise ValueError("S'u gjet tekst artikulli në këtë faqe.")
    return extracted.strip()


def render_result(label, confidence, all_probs, model_name):
    css_class = "result-fake" if label == "FAKE" else "result-real"
    title_class = "result-title-fake" if label == "FAKE" else "result-title-real"
    icon = "⚠️" if label == "FAKE" else "✅"
    st.markdown(
        f"""
        <div class="result-card {css_class}">
            <p class="{title_class}">{icon} {label} — {confidence*100:.1f}% siguri</p>
            <p class="result-sub">Modeli: {model_name}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.progress(confidence)
    c1, c2 = st.columns(2)
    c1.metric("Probabiliteti REAL", f"{all_probs[0]*100:.1f}%")
    c2.metric("Probabiliteti FAKE", f"{all_probs[1]*100:.1f}%")


# ---------------------------------------------------------------------------
# TAB 1: ANALIZO ARTIKULL
# ---------------------------------------------------------------------------
with tab_analyze:
    with st.sidebar:
        st.header("⚙️ Cilësimet")
        model_choice = st.selectbox("Zgjidh modelin", list(MODELS.keys()))
        st.caption(f"`{MODELS[model_choice]['description']}`")
        st.markdown("---")
        st.markdown(
            "**Si funksionon:** modeli lexon tekstin e artikullit dhe jep parashikimin "
            "Fake/Real, bashkë me një shkallë sigurie (confidence)."
        )

    with st.container(border=True):
        st.markdown('<div class="section-title">✨ Shembuj të shpejtë</div>', unsafe_allow_html=True)
        ex_cols = st.columns(len(EXAMPLES))
        for i, (ex_name, ex_text) in enumerate(EXAMPLES.items()):
            if ex_cols[i].button(ex_name, use_container_width=True):
                st.session_state["input_text"] = ex_text
                st.session_state["fetched_text"] = ex_text

    st.write("")

    with st.container(border=True):
        st.markdown('<div class="section-title">📝 Artikulli</div>', unsafe_allow_html=True)
        input_mode = st.radio(
            "Si do ta japësh lajmin?", ["Ngjit tekstin", "Vendos link (URL)"],
            horizontal=True, label_visibility="collapsed",
        )

        if input_mode == "Ngjit tekstin":
            text = st.text_area(
                "Ngjit tekstin e një lajmi në shqip:",
                value=st.session_state.get("input_text", ""),
                height=180,
                placeholder="P.sh. ngjit titullin dhe përmbajtjen e një artikulli lajmesh...",
                key="text_input_area",
            )
        else:
            url = st.text_input("Vendos linkun e artikullit:", placeholder="https://...")
            if st.button("Merr artikullin nga linku"):
                if not url.strip():
                    st.warning("Fut një link para se të vazhdosh.")
                else:
                    try:
                        with st.spinner("Duke shkarkuar..."):
                            fetched = fetch_article_text(url.strip())
                        st.session_state["fetched_text"] = fetched
                        st.success(f"U morën {len(fetched)} karaktere.")
                    except ValueError as e:
                        st.error(str(e))
            text = st.text_area(
                "Teksti i marrë (mund ta redaktosh):",
                value=st.session_state.get("fetched_text", ""),
                height=180,
            )

        col1, col2 = st.columns([1, 1])
        analyze = col1.button("🔍 Analizo", type="primary", use_container_width=True)
        compare = col2.button("⚖️ Krahaso të dy modelet", use_container_width=True)

    if analyze or compare:
        if not text.strip():
            st.warning("Fut një tekst para se të analizosh.")
        elif compare:
            st.markdown("#### Rezultati i krahasimit")
            c1, c2 = st.columns(2)
            for col, (name, info) in zip((c1, c2), MODELS.items()):
                with col:
                    try:
                        label, confidence, all_probs = predict(text, info["path"])
                        render_result(label, confidence, all_probs, name)
                    except OSError:
                        st.error(f"S'u gjet modeli `{info['path']}`.")
        else:
            model_path = MODELS[model_choice]["path"]
            try:
                label, confidence, all_probs = predict(text, model_path)
                render_result(label, confidence, all_probs, model_choice)
                with st.expander("Detaje shtesë"):
                    st.write(f"**Modeli i përdorur:** {model_choice}")
                    st.write("**Gjatësia maksimale e analizuar:** 256 tokenë")
            except OSError:
                st.error(
                    f"S'u gjet modeli te `{model_path}`. Kontrollo variablën MODELS "
                    "në krye të app.py."
                )

    st.markdown("---")
    st.caption(
        "Kujdes: ky është një prototip akademik për demonstrim, jo një mjet i verifikuar "
        "për përdorim në prodhim. Rezultatet duhen interpretuar me kujdes."
    )

# ---------------------------------------------------------------------------
# TAB 2: GJETJET E KERKIMIT
# ---------------------------------------------------------------------------
with tab_findings:
    with st.container(border=True):
        st.markdown('<div class="section-title">🧪 Përmbledhje e eksperimenteve</div>', unsafe_allow_html=True)
        st.markdown(
            "Tri modele shumëgjuhëshe (mBERT, XLM-R, mT5) u testuan në dy skenarë: "
            "**zero-shot** (trajnim vetëm në anglisht, testim në shqip) dhe **few-shot** "
            "(trajnim anglisht + 2,772 shembuj shqip). U përfshi edhe një model bazë "
            "(TF-IDF + Logistic Regression) si pikë krahasimi."
        )
        results_df = pd.DataFrame({
            "Model": ["Baseline (TF-IDF+LR)", "mBERT", "XLM-R", "mT5"],
            "Zero-shot (AL)": [None, 67.2, 50.0, 64.8],
            "Few-shot (AL)": [89.6, 91.1, 95.5, 91.6],
        })
        st.dataframe(results_df, hide_index=True, use_container_width=True)
        st.caption("Vlerat në % (accuracy mbi 594 artikuj testimi shqip).")

    st.write("")

    with st.container(border=True):
        st.markdown('<div class="section-title">📏 Bias sipas gjatësisë së artikullit</div>', unsafe_allow_html=True)
        st.caption("XLM-R, model few-shot")
        st.markdown(
            "Artikujt shqip u ndanë në tre grupe sipas numrit të fjalëve. Modeli gabon "
            "dukshëm më shumë (False Negative Rate më e lartë) te artikujt e gjatë — "
            "lajme të rreme më të gjata e më të sofistikuara mbeten shpesh të paidentifikuara."
        )
        bias_df = pd.DataFrame({
            "Grupi": ["Shkurt", "Mesatar", "Gjatë"],
            "False Negative Rate (%)": [2.76, 6.67, 21.88],
            "False Positive Rate (%)": [1.79, 2.67, 3.01],
        }).set_index("Grupi")
        st.bar_chart(bias_df, color=["#7c3aed", "#c4b5fd"])

    st.write("")

    with st.container(border=True):
        st.markdown('<div class="section-title">⚖️ Drejtësia ndër-gjuhësore (zero-shot)</div>', unsafe_allow_html=True)
        st.markdown(
            "XLM-R duket \"më i drejtë\" sipas Equal Opportunity Gap (0.003), por kjo është "
            "artificiale — modeli kishte kolapsuar duke parashikuar \"fake\" për çdo artikull. "
            "mBERT (EOG 0.212) dhe mT5 (EOG 0.488) pasqyrojnë hendekë realë EN→AL."
        )
        st.info(
            "Për detaje të plota metodologjike dhe diskutim kritik, shih Kapitujt III–V "
            "të punimit të diplomës."
        )
