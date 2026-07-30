"""
Verifiko - Zbulues i Lajmeve te Rreme (Fake News Detector)
Almira Mecaj, Diplome ML - "Evaluating Bias and Fairness of Multilingual
Models on Albanian Fake News"
 
Kjo verzion shton nje shtrese "AI ruling" (Gemini) mbi klasifikuesin XLM-R/mBERT:
- Truth Score  -> direkt nga modeli i trajnuar (REAL confidence) - baza akademike e diplomes
- Reliability / Consensus / Impact -> vleresim kontekstual nga Gemini (LLM), jo nga modeli klasifikues
- Key Findings, Debate Comments, Verdikti final -> tekst i gjeneruar nga Gemini
"""
 
import json
import pathlib
import re
 
import pandas as pd
import requests
import streamlit as st
import torch
import trafilatura
from transformers import AutoTokenizer, AutoModelForSequenceClassification
 
# ---------------------------------------------------------------------------
# PWA HEAD TAGS (manifest, ikone, iOS meta) - shkruhen direkt ne index.html
# te vete paketes streamlit ne disk, nje here, kur nis app-i.
# ---------------------------------------------------------------------------
def _install_pwa_head_tags():
    try:
        index_path = pathlib.Path(st.__path__[0]) / "static" / "index.html"
        html = index_path.read_text(encoding="utf-8")
        marker = "<!-- verifiko-pwa-tags -->"
        if marker in html:
            return
        manifest_url = (
            "https://raw.githubusercontent.com/almiramecaj947-bot/"
            "fake-news-detector-shqip/main/static/manifest.json"
        )
        icon_url = (
            "https://raw.githubusercontent.com/almiramecaj947-bot/"
            "fake-news-detector-shqip/main/static/icon-192.png"
        )
        tags = (
            marker + "\n"
            f'<link rel="manifest" href="{manifest_url}">\n'
            f'<link rel="apple-touch-icon" href="{icon_url}">\n'
            '<meta name="theme-color" content="#0b0b12">\n'
            '<meta name="apple-mobile-web-app-capable" content="yes">\n'
            '<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">\n'
            '<meta name="apple-mobile-web-app-title" content="Verifiko">\n'
        )
        html = html.replace("</head>", tags + "</head>")
        index_path.write_text(html, encoding="utf-8")
    except Exception:
        pass
 
 
_install_pwa_head_tags()
 
# ---------------------------------------------------------------------------
# KONFIGURIMI I MODELEVE (klasifikuesi - baza akademike)
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
 
GEMINI_MODEL_CANDIDATES = ["gemini-flash-latest", "gemini-3.5-flash-lite", "gemini-flash-lite-latest"]
 
# ---------------------------------------------------------------------------
# GEMINI - "AI ruling" mbi rezultatin e klasifikuesit
# ---------------------------------------------------------------------------
def _get_gemini_key():
    try:
        return st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        return ""
 
 
def _extract_json(raw_text: str):
    """Gemini nganjehere e mbeshtjell JSON-in me ```json ... ``` - e pastrojme."""
    cleaned = raw_text.strip()
    cleaned = re.sub(r"^```(json)?", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    return json.loads(cleaned)
 
 
def gemini_ruling(article_text: str, label: str, confidence: float) -> dict:
    """Thirr Gemini per te prodhuar Reliability/Consensus/Impact + tekstin e
    analizes, gjetjeve kryesore, komenteve te 'debatit', dhe verdiktit final.
    Kthen nje dict me vlera default nese Gemini s'eshte i konfiguruar ose deshton."""
    fallback = {
        "reliability_score": 50,
        "consensus_score": 50,
        "impact_score": 50,
        "analysis_summary": "Analiza e detajuar me AI s'është e disponueshme (mungon çelësi Gemini te 'Secrets').",
        "key_findings": [
            {"tag": "Logjika", "text": "Aktivizo Gemini API te Secrets për gjetje të detajuara."},
            {"tag": "Ekzagjerim", "text": "—"},
            {"tag": "Evidencë", "text": "—"},
        ],
        "debate_supportive": {"author": "Lexuesi A", "text": "—"},
        "debate_critical": {"author": "Lexuesi B", "text": "—"},
        "verdict": "Vlerësimi bazohet vetëm te modeli klasifikues (shih Truth Score).",
    }
 
    api_key = _get_gemini_key()
    if not api_key:
        return fallback
 
    prompt = f"""Je një asistent i verifikimit të fakteve për një aplikacion demo diplome në shqip
("Verifiko"). Modeli i mësimit të makinës ka klasifikuar tekstin e mëposhtëm si "{label}"
me {confidence*100:.1f}% siguri. Analizo vetë tekstin dhe kthe VETËM një objekt JSON (asnjë
tekst tjetër, pa ```), me këtë strukturë të saktë:
 
{{
  "reliability_score": <numër 0-100, sa i besueshëm duket burimi/stili i shkrimit>,
  "consensus_score": <numër 0-100, sa përputhet pretendimi me atë çka dihet/raportohet zakonisht>,
  "impact_score": <numër 0-100, sa i rrezikshëm/dëmshëm do të ishte nëse besohej dhe ky pretendim është i rremë>,
  "analysis_summary": "<2-3 fjali shqip, përmbledhje objektive e analizës>",
  "key_findings": [
    {{"tag": "Logjika", "text": "<1 fjali>"}},
    {{"tag": "Ekzagjerim", "text": "<1 fjali>"}},
    {{"tag": "Evidencë", "text": "<1 fjali>"}}
  ],
  "debate_supportive": {{"author": "<emër i shpikur>", "text": "<1-2 fjali që mbrojnë/besojnë lajmin>"}},
  "debate_critical": {{"author": "<emër i shpikur>", "text": "<1-2 fjali skeptike ndaj lajmit>"}},
  "verdict": "<1 paragraf shqip, në stil 'vendimi final i gjyqit', objektiv, bazuar në logjikë>"
}}
 
Teksti i lajmit:
\"\"\"{article_text[:4000]}\"\"\"
"""
 
    for model_name in GEMINI_MODEL_CANDIDATES:
        try:
            resp = requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent",
                params={"key": api_key},
                json={
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {"responseMimeType": "application/json", "temperature": 0.4},
                },
                timeout=25,
            )
            if resp.status_code != 200:
                continue
            data = resp.json()
            raw_text = data["candidates"][0]["content"]["parts"][0]["text"]
            parsed = _extract_json(raw_text)
            for key, default in fallback.items():
                parsed.setdefault(key, default)
            return parsed
        except Exception:
            continue
    return fallback
 
 
# ---------------------------------------------------------------------------
# MODELI KLASIFIKUES
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
 
 
# ---------------------------------------------------------------------------
# UI SETUP - TEMA E ERRET
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Verifiko — Zbulues i Lajmeve të Rreme", page_icon="🛡️", layout="centered")
 
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@600;700;800&family=Inter:wght@400;500;600;700&display=swap');
 
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp { background: #0b0b12; }
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stToolbar"] { display: none; }
    .block-container { padding-top: 1.6rem; max-width: 760px; }
 
    h1, h2, h3, p, span, label, div { color: #e8e8ef; }
 
    .hero {
        display: flex; align-items: center; gap: 1rem;
        padding: 1.7rem 1.7rem; border-radius: 24px; margin-bottom: 1.1rem;
        background: linear-gradient(135deg, #6d28d9 0%, #5b21b6 55%, #2e1065 100%);
        box-shadow: 0 16px 40px rgba(0,0,0,0.45);
    }
    .hero-icon {
        flex-shrink: 0; width: 56px; height: 56px; border-radius: 16px;
        background: rgba(255,255,255,0.16); display: flex; align-items: center;
        justify-content: center; font-size: 1.8rem;
        border: 1px solid rgba(255,255,255,0.25);
    }
    .hero-text h1 {
        font-family: 'Poppins', sans-serif; margin: 0; font-size: 1.55rem;
        font-weight: 800; color: #ffffff !important; letter-spacing: -0.3px;
    }
    .hero-text p { margin: 0.3rem 0 0 0; color: rgba(255,255,255,0.85) !important; font-size: 0.92rem; line-height: 1.4; }
 
    .pill-row { display: flex; gap: 0.5rem; margin-bottom: 1.4rem; flex-wrap: wrap; }
    .pill {
        display: inline-flex; align-items: center; gap: 0.35rem;
        padding: 0.4rem 0.9rem; border-radius: 999px; font-size: 0.8rem; font-weight: 600;
        background: rgba(124,58,237,0.18); color: #c4b5fd !important; border: 1px solid rgba(124,58,237,0.35);
    }
 
    .app-card {
        background: #17171f; border: 1px solid #26262f; border-radius: 20px;
        padding: 1.3rem 1.4rem; margin-bottom: 1rem; box-shadow: 0 4px 18px rgba(0,0,0,0.25);
    }
    .app-card-title {
        font-family: 'Poppins', sans-serif; font-weight: 700; font-size: 1rem;
        color: #f4f4f7 !important; margin-bottom: 0.7rem; display: flex; align-items: center; gap: 0.45rem;
    }
 
    div[role="radiogroup"] { gap: 0.5rem; }
 
    .stButton > button {
        border-radius: 12px !important; font-weight: 600 !important; border: 1px solid #2e2e3a !important;
        background: #1e1e28 !important; color: #d4d4dc !important;
    }
    .stButton > button:hover { border-color: #7c3aed !important; color: #c4b5fd !important; }
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #7c3aed, #5b21b6) !important;
        color: #ffffff !important; border: none !important;
        box-shadow: 0 8px 20px rgba(91,33,182,0.45) !important;
    }
    .stButton > button[kind="primary"]:hover { filter: brightness(1.1); color: #ffffff !important; }
 
    [data-baseweb="tab-list"] { gap: 0.4rem; background: #17171f; padding: 0.35rem; border-radius: 14px; }
    [data-baseweb="tab"] { border-radius: 10px !important; font-weight: 600; color: #9ca3af !important; }
    [aria-selected="true"][data-baseweb="tab"] {
        background: #2a2a38 !important; color: #c4b5fd !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.25);
    }
 
    textarea, input { background: #101018 !important; color: #e8e8ef !important; border-color: #2e2e3a !important; }
 
    /* ---------- 4 GAUGE SCORE CARDS ---------- */
    .score-row { display: flex; gap: 0.7rem; margin: 0.8rem 0 1rem 0; flex-wrap: wrap; }
    .score-card {
        flex: 1; min-width: 140px; background: #1c1c26; border: 1px solid #292935;
        border-radius: 16px; padding: 0.9rem 0.7rem; text-align: center;
    }
    .score-card-label { font-size: 0.78rem; color: #9ca3af !important; font-weight: 600; margin-bottom: 0.15rem; }
    .score-card-sub { font-size: 0.68rem; color: #6b7280 !important; margin-bottom: 0.6rem; min-height: 1.4em; }
    .score-gauge {
        width: 68px; height: 68px; border-radius: 50%; margin: 0 auto;
        display: flex; align-items: center; justify-content: center;
    }
    .score-gauge-inner {
        width: 54px; height: 54px; border-radius: 50%; background: #1c1c26;
        display: flex; align-items: center; justify-content: center;
        font-weight: 800; font-size: 0.95rem; font-family: 'Poppins', sans-serif;
    }
 
    /* ---------- KEY FINDINGS ---------- */
    .finding-card {
        background: #1c1c26; border: 1px solid #292935; border-radius: 14px;
        padding: 0.8rem 1rem; margin-bottom: 0.6rem;
    }
    .finding-tag {
        display: inline-block; font-size: 0.68rem; font-weight: 700; text-transform: uppercase;
        color: #c4b5fd !important; background: rgba(124,58,237,0.18); padding: 0.15rem 0.6rem;
        border-radius: 999px; margin-bottom: 0.4rem; letter-spacing: 0.03em;
    }
    .finding-text { color: #d4d4dc !important; font-size: 0.9rem; line-height: 1.4; }
 
    /* ---------- DEBATE COMMENTS ---------- */
    .debate-bubble {
        border-radius: 14px; padding: 0.8rem 1rem; margin-bottom: 0.6rem; border: 1px solid;
    }
    .debate-bubble.supportive { background: rgba(59,130,246,0.10); border-color: rgba(59,130,246,0.35); }
    .debate-bubble.critical { background: rgba(239,68,68,0.10); border-color: rgba(239,68,68,0.35); }
    .debate-author { font-weight: 700; font-size: 0.82rem; margin-bottom: 0.2rem; }
    .debate-bubble.supportive .debate-author { color: #93c5fd !important; }
    .debate-bubble.critical .debate-author { color: #fca5a5 !important; }
    .debate-text { color: #d4d4dc !important; font-size: 0.88rem; line-height: 1.4; }
 
    /* ---------- VERDICT CARD ---------- */
    .final-verdict-card {
        background: linear-gradient(135deg, rgba(124,58,237,0.18), rgba(91,33,182,0.10));
        border: 1px solid rgba(124,58,237,0.35); border-radius: 18px;
        padding: 1.2rem 1.3rem; margin-top: 0.4rem;
    }
    .final-verdict-title {
        font-family: 'Poppins', sans-serif; font-weight: 800; font-size: 1rem;
        color: #c4b5fd !important; margin-bottom: 0.6rem;
    }
    .final-verdict-text { color: #e8e8ef !important; font-size: 0.92rem; line-height: 1.55; }
 
    .disclaimer {
        font-size: 0.78rem; color: #6b7280 !important; text-align: center; margin-top: 1.4rem; line-height: 1.4;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
 
 
def gauge_color(pct: float, invert: bool = False) -> str:
    """E kuqe per rrezik/pasiguri te larte, jeshile per mire. invert=True per
    metrika ku numer i larte = keq (p.sh. Impact)."""
    v = (100 - pct) if invert else pct
    if v >= 70:
        return "#22c55e"
    if v >= 40:
        return "#f59e0b"
    return "#ef4444"
 
 
def score_card(label: str, sub: str, pct: float, invert: bool = False):
    color = gauge_color(pct, invert=invert)
    deg = max(0, min(100, pct)) * 3.6
    st.markdown(
        f"""
        <div class="score-card">
            <div class="score-card-label">{label}</div>
            <div class="score-card-sub">{sub}</div>
            <div class="score-gauge" style="background: conic-gradient({color} {deg}deg, #2a2a38 0deg);">
                <div class="score-gauge-inner" style="color:{color};">{pct:.0f}%</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
 
 
# ---------------------------------------------------------------------------
# ONBOARDING (shfaqet 1 here per sesion)
# ---------------------------------------------------------------------------
ONBOARDING_SLIDES = [
    ("🕵️", "Zbulo Bias-in dhe Axhendat e Fshehura",
     "Verifiko analizon gjuhën dhe tonin e artikullit për sinjale anësie apo manipulimi."),
    ("🧠", "Analiza me Inteligjencë Artificiale",
     "Modeli XLM-R (95.5% saktësi) klasifikon tekstin, ndërsa Gemini shpjegon logjikën pas rezultatit."),
    ("⚖️", "Pikëpamje e Balancuar",
     "Shiko argumentet mbështetëse dhe kritike krah për krah, para se të nxjerrësh përfundimin tënd."),
    ("🔨", "Vendimi Final",
     "Merr një 'verdikt' objektiv, bazuar në logjikë dhe evidencë — jo emocion."),
]
 
if "onboarded" not in st.session_state:
    st.session_state["onboarded"] = False
if "onb_step" not in st.session_state:
    st.session_state["onb_step"] = 0
 
if not st.session_state["onboarded"]:
    step = st.session_state["onb_step"]
    icon, title, desc = ONBOARDING_SLIDES[step]
    st.markdown(
        f"""
        <div class="hero" style="flex-direction: column; text-align: center; padding: 2.6rem 1.6rem;">
            <div class="hero-icon" style="font-size: 2.6rem; width: 84px; height: 84px; margin-bottom: 1rem;">{icon}</div>
            <div class="hero-text">
                <h1 style="font-size: 1.5rem;">{title}</h1>
                <p style="font-size: 1rem; margin-top: 0.6rem;">{desc}</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    dots = " ".join("●" if i == step else "○" for i in range(len(ONBOARDING_SLIDES)))
    st.markdown(f"<p style='text-align:center; color:#7c3aed; letter-spacing:0.3em;'>{dots}</p>", unsafe_allow_html=True)
 
    c1, c2 = st.columns([1, 1])
    if step < len(ONBOARDING_SLIDES) - 1:
        if c1.button("Kapërce", use_container_width=True):
            st.session_state["onboarded"] = True
            st.rerun()
        if c2.button("Tjetra →", type="primary", use_container_width=True):
            st.session_state["onb_step"] += 1
            st.rerun()
    else:
        if c2.button("Fillo ✓", type="primary", use_container_width=True):
            st.session_state["onboarded"] = True
            st.rerun()
    st.stop()
 
# ---------------------------------------------------------------------------
# HERO (pas onboarding)
# ---------------------------------------------------------------------------
st.markdown(
    """
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
        <span class="pill">🤖 XLM-R + Gemini</span>
        <span class="pill">🇦🇱 Shqip · 🇬🇧 Anglisht</span>
    </div>
    """,
    unsafe_allow_html=True,
)
 
tab_analyze, tab_findings = st.tabs(["🔍  Analizo", "📊  Gjetjet e Kërkimit"])
 
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
 
    analyze = st.button("🔍 Analizo", type="primary", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
 
    if analyze:
        if not text.strip():
            st.warning("Fut një tekst para se të analizosh.")
        else:
            model_path = MODELS[model_choice]["path"]
            try:
                label, confidence, all_probs = predict(text, model_path)
                truth_pct = all_probs[0] * 100  # probabiliteti REAL = "Truth Score"
 
                with st.spinner("Duke analizuar me AI..."):
                    ruling = gemini_ruling(text, label, confidence)
 
                result_tab_analysis, result_tab_verdict = st.tabs(["📋 Analiza", "⚖️ Verdikti"])
 
                with result_tab_analysis:
                    st.markdown('<div class="score-row">', unsafe_allow_html=True)
                    cols = st.columns(4)
                    with cols[0]:
                        score_card("Truth Score", f"Modeli {model_choice}", truth_pct)
                    with cols[1]:
                        score_card("Besueshmëria", "Stili/toni", float(ruling.get("reliability_score", 50)))
                    with cols[2]:
                        score_card("Konsensusi", "Vs. raportime tjera", float(ruling.get("consensus_score", 50)))
                    with cols[3]:
                        score_card("Ndikimi", "Rrezik nëse i rremë", float(ruling.get("impact_score", 50)), invert=True)
                    st.markdown('</div>', unsafe_allow_html=True)
 
                    st.markdown('<div class="app-card">', unsafe_allow_html=True)
                    st.markdown('<div class="app-card-title">📄 Përmbledhje e Analizës</div>', unsafe_allow_html=True)
                    st.markdown(f'<p class="finding-text">{ruling.get("analysis_summary", "")}</p>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
 
                    st.markdown('<div class="app-card">', unsafe_allow_html=True)
                    st.markdown('<div class="app-card-title">🔑 Gjetjet Kryesore</div>', unsafe_allow_html=True)
                    for finding in ruling.get("key_findings", []):
                        st.markdown(
                            f"""<div class="finding-card">
                                <span class="finding-tag">{finding.get('tag','')}</span>
                                <div class="finding-text">{finding.get('text','')}</div>
                            </div>""",
                            unsafe_allow_html=True,
                        )
                    st.markdown('</div>', unsafe_allow_html=True)
 
                    st.markdown('<div class="app-card">', unsafe_allow_html=True)
                    st.markdown('<div class="app-card-title">💬 Debati</div>', unsafe_allow_html=True)
                    sup = ruling.get("debate_supportive", {})
                    crit = ruling.get("debate_critical", {})
                    st.markdown(
                        f"""<div class="debate-bubble supportive">
                            <div class="debate-author">{sup.get('author','')} · Mbështetës</div>
                            <div class="debate-text">{sup.get('text','')}</div>
                        </div>""",
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f"""<div class="debate-bubble critical">
                            <div class="debate-author">{crit.get('author','')} · Kritik</div>
                            <div class="debate-text">{crit.get('text','')}</div>
                        </div>""",
                        unsafe_allow_html=True,
                    )
                    st.markdown('</div>', unsafe_allow_html=True)
 
                with result_tab_verdict:
                    icon = "⚠️" if label == "FAKE" else "✅"
                    verdict_label = "LAJM I RREMË" if label == "FAKE" else "LAJM I BESUESHËM"
                    st.markdown(
                        f"""
                        <div class="final-verdict-card">
                            <div class="final-verdict-title">{icon} {verdict_label} — Truth Score {truth_pct:.0f}%</div>
                            <div class="final-verdict-text">{ruling.get("verdict", "")}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.caption(f"Klasifikuesi {model_choice}: {label} me {confidence*100:.1f}% siguri.")
 
            except OSError:
                st.error(
                    f"S'u gjet modeli te `{model_path}`. Kontrollo variablën MODELS "
                    "në krye të app.py."
                )
 
    st.markdown(
        '<p class="disclaimer">Kujdes: ky është një prototip akademik për demonstrim, jo një mjet '
        'i verifikuar për përdorim në prodhim. Rezultatet (përfshi ato të gjeneruara nga AI) '
        'duhen interpretuar me kujdes.</p>',
        unsafe_allow_html=True,
    )
 
# ---------------------------------------------------------------------------
# TAB 2: GJETJET E KERKIMIT (permbajtje akademike, e paprekur)
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
 
