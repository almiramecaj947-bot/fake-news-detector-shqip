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
    'i instalueshem' (Add to Home Screen / APK me PWABuilder), si nje app real."""
    components.html(
        """
        <script>
        (function() {
            const parentDoc = window.parent.document;
            // Streamlit Cloud's own static-file serving proved unreliable, so the
            // manifest + icons are hosted directly on GitHub's raw CDN instead.
            const manifestUrl = 'https://raw.githubusercontent.com/almiramecaj947-bot/fake-news-detector-shqip/main/static/manifest.json';
            const iconUrl = 'https://raw.githubusercontent.com/almiramecaj947-bot/fake-news-detector-shqip/main/static/icon-192.png';
 
            // Streamlit ships its own default manifest link — override it (don't
            // just skip if one exists, or our manifest never wins).
            let manifestLink = parentDoc.querySelector('link[rel="manifest"]');
            if (!manifestLink) {
                manifestLink = parentDoc.createElement('link');
                manifestLink.rel = 'manifest';
                parentDoc.head.appendChild(manifestLink);
            }
            manifestLink.href = manifestUrl;
 
            let appleIcon = parentDoc.querySelector('link[rel="apple-touch-icon"]');
            if (!appleIcon) {
                appleIcon = parentDoc.createElement('link');
                appleIcon.rel = 'apple-touch-icon';
                parentDoc.head.appendChild(appleIcon);
            }
            appleIcon.href = iconUrl;
 
            let themeMeta = parentDoc.querySelector('meta[name="theme-color"]');
            if (!themeMeta) {
                themeMeta = parentDoc.createElement('meta');
                themeMeta.name = 'theme-color';
                parentDoc.head.appendChild(themeMeta);
            }
            themeMeta.content = '#5b21b6';
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
    "XLM-R": {
        "path": "almira123/xlmr-albanian-fake-news",
        "description": "xlm-roberta-base — fine-tuned EN + AL",
        "accuracy": "95.5%",
    },
    "mBERT": {
        "path": "almira123/mbert-albanian-fake-news",
        "description": "bert-base-multilingual-cased — fine-tuned EN + AL",
        "accuracy": "91.1%",
    },
}
 
LABELS = {0: "REAL", 1: "FAKE"}
 
EXAMPLES = {
    "📗  Shembull real": (
        "Adelina e tepron me fustanin e shkurtër Adelina Tahiri është një ndër femrat më "
        "provokuese në mediat rozë. Duke mos hezituar që të pozojë në forma të ndryshme, "
        "këngëtarja duket gjithmonë e më e zjarrtë me stilin e veçantë që ka. Sidomos, në "
        "imazhin e fundit me një fustan të shkurtër dhe të ngushtë ajo e teproi me pozën që "
        "ka realizuar. Theksojmë, ajo kohëve të fundit mungon në projekte muzikore."
    ),
    "📕  Shembull i rremë": (
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
st.set_page_config(page_title="Verifiko — Zbulues i Lajmeve të Rreme", page_icon="🛡️", layout="centered")
inject_pwa_head()
 
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@600;700;800&family=Inter:wght@400;500;600;700&display=swap');
 
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
 
    /* hide default streamlit chrome for an app-like feel */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stToolbar"] { display: none; }
    .block-container { padding-top: 1.6rem; max-width: 760px; }
 
    /* ---------- HERO ---------- */
    .hero {
        display: flex; align-items: center; gap: 1rem;
        padding: 1.7rem 1.7rem; border-radius: 24px; margin-bottom: 1.1rem;
        background: linear-gradient(135deg, #6d28d9 0%, #5b21b6 55%, #4c1d95 100%);
        box-shadow: 0 16px 40px rgba(91,33,182,0.35);
    }
    .hero-icon {
        flex-shrink: 0; width: 56px; height: 56px; border-radius: 16px;
        background: rgba(255,255,255,0.16); display: flex; align-items: center;
        justify-content: center; font-size: 1.8rem;
        border: 1px solid rgba(255,255,255,0.25);
    }
    .hero-text h1 {
        font-family: 'Poppins', sans-serif; margin: 0; font-size: 1.55rem;
        font-weight: 800; color: #ffffff; letter-spacing: -0.3px;
    }
    .hero-text p { margin: 0.3rem 0 0 0; color: rgba(255,255,255,0.85); font-size: 0.92rem; line-height: 1.4; }
 
    .pill-row { display: flex; gap: 0.5rem; margin-bottom: 1.4rem; flex-wrap: wrap; }
    .pill {
        display: inline-flex; align-items: center; gap: 0.35rem;
        padding: 0.4rem 0.9rem; border-radius: 999px; font-size: 0.8rem; font-weight: 600;
        background: #f3f0ff; color: #5b21b6; border: 1px solid #e4d9ff;
    }
 
    /* ---------- SECTION CARD ---------- */
    .app-card {
        background: #ffffff; border: 1px solid #ececf3; border-radius: 20px;
        padding: 1.3rem 1.4rem; margin-bottom: 1rem; box-shadow: 0 4px 18px rgba(17,24,39,0.05);
    }
    .app-card-title {
        font-family: 'Poppins', sans-serif; font-weight: 700; font-size: 1rem;
        color: #1f2937; margin-bottom: 0.7rem; display: flex; align-items: center; gap: 0.45rem;
    }
 
    /* ---------- MODEL PICKER (segmented) ---------- */
    div[role="radiogroup"] { gap: 0.5rem; }
 
    /* buttons — set bg + text color together always, avoid white-on-white regressions */
    .stButton > button {
        border-radius: 12px !important; font-weight: 600 !important; border: 1px solid #e5e7eb !important;
        background: #f9fafb !important; color: #374151 !important;
    }
    .stButton > button:hover { border-color: #c4b5fd !important; color: #5b21b6 !important; }
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #7c3aed, #5b21b6) !important;
        color: #ffffff !important; border: none !important;
        box-shadow: 0 8px 20px rgba(91,33,182,0.30) !important;
    }
    .stButton > button[kind="primary"]:hover { filter: brightness(1.06); color: #ffffff !important; }
 
    /* tabs -> pill nav */
    [data-baseweb="tab-list"] { gap: 0.4rem; background: #f4f4f8; padding: 0.35rem; border-radius: 14px; }
    [data-baseweb="tab"] {
        border-radius: 10px !important; font-weight: 600; color: #6b7280;
    }
    [aria-selected="true"][data-baseweb="tab"] {
        background: #ffffff !important; color: #5b21b6 !important;
        box-shadow: 0 2px 8px rgba(17,24,39,0.08);
    }
 
    /* ---------- VERDICT CARD ---------- */
    .verdict-card {
        display: flex; align-items: center; gap: 1.2rem; border-radius: 20px;
        padding: 1.3rem 1.4rem; margin-top: 0.6rem; border: 1px solid; box-shadow: 0 6px 20px rgba(17,24,39,0.06);
    }
    .verdict-gauge {
        flex-shrink: 0; width: 92px; height: 92px; border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
    }
    .verdict-gauge-inner {
        width: 70px; height: 70px; border-radius: 50%; background: #ffffff;
        display: flex; align-items: center; justify-content: center;
        font-family: 'Poppins', sans-serif; font-weight: 800; font-size: 1.05rem;
    }
    .verdict-label { font-family: 'Poppins', sans-serif; font-weight: 800; font-size: 1.15rem; }
    .verdict-sub { color: #6b7280; font-size: 0.85rem; margin: 0.15rem 0 0.7rem 0; }
    .vbar-row { display: flex; align-items: center; gap: 0.5rem; font-size: 0.78rem; color: #4b5563; margin-bottom: 0.3rem; }
    .vbar-row span:first-child { width: 34px; }
    .vbar-row span:last-child { width: 34px; text-align: right; }
    .vbar-track { flex: 1; height: 7px; border-radius: 999px; background: #eef0f4; overflow: hidden; }
    .vbar-fill { height: 100%; border-radius: 999px; }
 
    .disclaimer {
        font-size: 0.78rem; color: #9ca3af; text-align: center; margin-top: 1.4rem; line-height: 1.4;
    }
    </style>
 
    <div class="hero">
        <div class="hero-icon">🛡️</div>
        <div class="hero-text">
            <h1>Verifiko</h1>
            <p>Zbulues i lajmeve të rreme në shqip — prototip praktik i punimit të diplomës
            "Evaluating Bias and Fairness of Multilingual Models on Albanian Fake News".</p>
        </div>
    </div>
    <div class="pill-row">
        <span class="pill">🎯 95.5% saktësi</span>
        <span class="pill">🤖 2 modele AI</span>
        <span class="pill">🇦🇱 Shqip · 🇬🇧 Anglisht</span>
    </div>
    """,
    unsafe_allow_html=True,
)
 
tab_analyze, tab_findings = st.tabs(["🔍  Analizo", "📊  Gjetjet"])
 
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
    is_fake = label == "FAKE"
    accent = "#dc2626" if is_fake else "#16a34a"
    accent_soft = "#fef2f2" if is_fake else "#f0fdf4"
    accent_border = "#fecaca" if is_fake else "#bbf7d0"
    icon = "⚠️" if is_fake else "✅"
    verdict_text = "LAJM I RREMË" if is_fake else "LAJM I BESUESHËM"
    pct = confidence * 100
    deg = pct * 3.6
    st.markdown(
        f"""
        <div class="verdict-card" style="background:{accent_soft}; border-color:{accent_border};">
            <div class="verdict-gauge" style="background: conic-gradient({accent} {deg}deg, #e5e7eb 0deg);">
                <div class="verdict-gauge-inner" style="color:{accent};">{pct:.0f}%</div>
            </div>
            <div style="flex:1;">
                <div class="verdict-label" style="color:{accent};">{icon} {verdict_text}</div>
                <div class="verdict-sub">Siguria e modelit {model_name} · {pct:.1f}%</div>
                <div class="vbar-row"><span>Real</span><div class="vbar-track"><div class="vbar-fill" style="width:{all_probs[0]*100:.1f}%; background:#16a34a;"></div></div><span>{all_probs[0]*100:.0f}%</span></div>
                <div class="vbar-row"><span>Fake</span><div class="vbar-track"><div class="vbar-fill" style="width:{all_probs[1]*100:.1f}%; background:#dc2626;"></div></div><span>{all_probs[1]*100:.0f}%</span></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
 
 
# ---------------------------------------------------------------------------
# TAB 1: ANALIZO
# ---------------------------------------------------------------------------
with tab_analyze:
    st.markdown('<div class="app-card">', unsafe_allow_html=True)
    st.markdown('<div class="app-card-title">🤖 Modeli</div>', unsafe_allow_html=True)
    model_choice = st.radio(
        "Zgjidh modelin", list(MODELS.keys()), horizontal=True, label_visibility="collapsed",
    )
    st.caption(f"{MODELS[model_choice]['description']} · saktësi few-shot: **{MODELS[model_choice]['accuracy']}**")
    st.markdown('</div>', unsafe_allow_html=True)
 
    st.markdown('<div class="app-card">', unsafe_allow_html=True)
    st.markdown('<div class="app-card-title">✨ Shembuj të shpejtë</div>', unsafe_allow_html=True)
    ex_cols = st.columns(len(EXAMPLES))
    for i, (ex_name, ex_text) in enumerate(EXAMPLES.items()):
        if ex_cols[i].button(ex_name, use_container_width=True):
            st.session_state["input_text"] = ex_text
            st.session_state["fetched_text"] = ex_text
    st.markdown('</div>', unsafe_allow_html=True)
 
    st.markdown('<div class="app-card">', unsafe_allow_html=True)
    st.markdown('<div class="app-card-title">📝 Artikulli</div>', unsafe_allow_html=True)
    input_mode = st.radio(
        "Si do ta japësh lajmin?", ["Ngjit tekstin", "Vendos link (URL)"],
        horizontal=True, label_visibility="collapsed",
    )
 
    if input_mode == "Ngjit tekstin":
        text = st.text_area(
            "Ngjit tekstin e një lajmi në shqip:",
            value=st.session_state.get("input_text", ""),
            height=170,
            placeholder="P.sh. ngjit titullin dhe përmbajtjen e një artikulli lajmesh...",
            key="text_input_area",
            label_visibility="collapsed",
        )
    else:
        url = st.text_input("Vendos linkun e artikullit:", placeholder="https://...", label_visibility="collapsed")
        if st.button("⬇️  Merr artikullin nga linku"):
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
            height=170,
        )
 
    col1, col2 = st.columns([1, 1])
    analyze = col1.button("🔍 Analizo", type="primary", use_container_width=True)
    compare = col2.button("⚖️ Krahaso modelet", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
 
    if analyze or compare:
        if not text.strip():
            st.warning("Fut një tekst para se të analizosh.")
        elif compare:
            st.markdown('<div class="app-card-title">⚖️ Rezultati i krahasimit</div>', unsafe_allow_html=True)
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
            except OSError:
                st.error(
                    f"S'u gjet modeli te `{model_path}`. Kontrollo variablën MODELS "
                    "në krye të app.py."
                )
 
    st.markdown(
        '<p class="disclaimer">Kujdes: ky është një prototip akademik për demonstrim, jo një mjet '
        'i verifikuar për përdorim në prodhim. Rezultatet duhen interpretuar me kujdes.</p>',
        unsafe_allow_html=True,
    )
 
# ---------------------------------------------------------------------------
# TAB 2: GJETJET E KERKIMIT
# ---------------------------------------------------------------------------
with tab_findings:
    st.markdown('<div class="app-card">', unsafe_allow_html=True)
    st.markdown('<div class="app-card-title">🧪 Përmbledhje e eksperimenteve</div>', unsafe_allow_html=True)
    st.markdown(
        "Katër modele (bazë TF-IDF+Logistic Regression, mBERT, XLM-R, mT5) u testuan në dy skenarë: "
        "**zero-shot** (trajnim vetëm në anglisht, testim në shqip) dhe **few-shot** "
        "(trajnim anglisht + 2,772 shembuj shqip)."
    )
    results_df = pd.DataFrame({
        "Model": ["Baseline (TF-IDF+LR)", "mBERT", "XLM-R", "mT5"],
        "Zero-shot (AL)": [None, 67.2, 50.0, 64.8],
        "Few-shot (AL)": [89.6, 91.1, 95.5, 91.6],
    })
    st.dataframe(results_df, hide_index=True, use_container_width=True)
    st.caption("Vlerat në % (accuracy mbi 594 artikuj testimi shqip).")
    st.markdown('</div>', unsafe_allow_html=True)
 
    st.markdown('<div class="app-card">', unsafe_allow_html=True)
    st.markdown('<div class="app-card-title">📏 Bias sipas gjatësisë së artikullit</div>', unsafe_allow_html=True)
    st.caption("XLM-R, model few-shot")
    st.markdown(
        "Artikujt shqip u ndanë në tre grupe sipas numrit të fjalëve. Modeli gabon "
        "dukshëm më shumë (False Negative Rate më e lartë) te artikujt e gjatë."
    )
    bias_df = pd.DataFrame({
        "Grupi": ["Shkurt", "Mesatar", "Gjatë"],
        "False Negative Rate (%)": [2.76, 6.67, 21.88],
        "False Positive Rate (%)": [1.79, 2.67, 3.01],
    }).set_index("Grupi")
    st.bar_chart(bias_df)
    st.markdown('</div>', unsafe_allow_html=True)
 
    st.markdown('<div class="app-card">', unsafe_allow_html=True)
    st.markdown('<div class="app-card-title">⚖️ Drejtësia ndër-gjuhësore (zero-shot)</div>', unsafe_allow_html=True)
    st.markdown(
        "XLM-R duket \"më i drejtë\" sipas Equal Opportunity Gap (0.003), por kjo është "
        "artificiale — modeli kishte kolapsuar duke parashikuar \"fake\" për çdo artikull. "
        "mBERT (EOG 0.212) dhe mT5 (EOG 0.488) pasqyrojnë hendekë realë EN→AL."
    )
    st.info("Për detaje të plota metodologjike, shih Kreun III–IV të punimit të diplomës.")
    st.markdown('</div>', unsafe_allow_html=True)
