"""
Verify Albanian News (Verifiko) - Zbulues i Lajmeve te Rreme
Almira Mecaj, Diplome ML - "Evaluating Bias and Fairness of Multilingual
Models on Albanian Fake News"

v10: menu fillestar (Analize e Detajuar / Verifikimi AI - chat), riorganizim
i faqes se analizes per te ndjekur nga afer nje app reference (headline +
scores direkt, gjetje, debat, verdikt me pjesemarres/rast/fakte te verifikuara).
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

APP_TITLE = "Verify Albanian News"

# ---------------------------------------------------------------------------
# PWA HEAD TAGS
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
    "Shembull real": (
        "Adelina e tepron me fustanin e shkurtër Adelina Tahiri është një ndër femrat më "
        "provokuese në mediat rozë. Duke mos hezituar që të pozojë në forma të ndryshme, "
        "këngëtarja duket gjithmonë e më e zjarrtë me stilin e veçantë që ka. Sidomos, në "
        "imazhin e fundit me një fustan të shkurtër dhe të ngushtë ajo e teproi me pozën që "
        "ka realizuar. Theksojmë, ajo kohëve të fundit mungon në projekte muzikore."
    ),
    "Shembull i rremë": (
        "Kjo është mundësia e ardhjes së mërgimtarëve nga Gjermania. Gazetari i Deutsche "
        "Welle, Bahri Cani, ka folur për mundësitë që kanë kosovarët të cilët jetojnë në "
        "Gjermani për të ardhur drejtë Kosovës për pushime verore. “500 mijë shqiptarë sa "
        "jetojnë në Gjermani dëshirojnë që pushimet e tyre t’i kalojnë në Kosovë, Shqipëri "
        "dhe vendet tjera” — shiko pamjet se si mund të udhëtojnë mërgimtarët për në vendlindje."
    ),
}

GEMINI_MODEL_CANDIDATES = ["gemini-flash-latest", "gemini-3.5-flash-lite", "gemini-flash-lite-latest"]


# ---------------------------------------------------------------------------
# GEMINI HELPERS
# ---------------------------------------------------------------------------
def _get_gemini_key():
    try:
        return st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        return ""


def _extract_json(raw_text: str):
    cleaned = raw_text.strip()
    cleaned = re.sub(r"^```(json)?", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    return json.loads(cleaned)


def _gemini_generate(contents, response_json=True, temperature=0.4):
    """Thirrje e pergjithshme Gemini. contents = lista Gemini-style [{'role':.., 'parts':[{'text':..}]}]."""
    api_key = _get_gemini_key()
    if not api_key:
        return None
    gen_config = {"temperature": temperature}
    if response_json:
        gen_config["responseMimeType"] = "application/json"
    for model_name in GEMINI_MODEL_CANDIDATES:
        try:
            resp = requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent",
                params={"key": api_key},
                json={"contents": contents, "generationConfig": gen_config},
                timeout=25,
            )
            if resp.status_code != 200:
                continue
            data = resp.json()
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except Exception:
            continue
    return None


def gemini_ruling(article_text: str, label: str, confidence: float) -> dict:
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
        "headline": article_text.strip().split("\n")[0][:90],
    }

    prompt = f"""Je një asistent i verifikimit të fakteve për një aplikacion demo diplome në shqip
("{APP_TITLE}"). Modeli i mësimit të makinës ka klasifikuar tekstin e mëposhtëm si "{label}"
me {confidence*100:.1f}% siguri. Analizo vetë tekstin dhe kthe VETËM një objekt JSON (asnjë
tekst tjetër, pa ```), me këtë strukturë të saktë:

{{
  "headline": "<titull i shkurtër (max 12 fjalë) që përmbledh temën e lajmit, në shqip>",
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
    raw_text = _gemini_generate([{"parts": [{"text": prompt}]}], response_json=True)
    if raw_text is None:
        return fallback
    try:
        parsed = _extract_json(raw_text)
        for key, default in fallback.items():
            parsed.setdefault(key, default)
        return parsed
    except Exception:
        return fallback


def gemini_chat_reply(history: list) -> str:
    """history = lista [{'role': 'user'|'assistant', 'text': ...}, ...]"""
    system_note = (
        "Je 'Verifikimi AI', asistenti bisedues i aplikacionit Verify Albanian News. "
        "Përgjigju gjithmonë në shqip, shkurt dhe qartë. Kur dikush të japë një lajm ose "
        "pretendim, vlerëso besueshmërinë e tij si do ta bënte një gazetar/fact-checker "
        "profesionist: shqyrto logjikën, ekzagjerimin, dhe evidencën e mundshme. Nëse pyetja "
        "s'ka lidhje me lajme/fakte, përgjigju normalisht por kthehu te roli yt kryesor."
    )
    contents = [{"role": "user", "parts": [{"text": system_note}]},
                {"role": "model", "parts": [{"text": "Kuptova. Jam gati të ndihmoj me verifikimin e lajmeve."}]}]
    for turn in history:
        role = "user" if turn["role"] == "user" else "model"
        contents.append({"role": role, "parts": [{"text": turn["text"]}]})

    reply = _gemini_generate(contents, response_json=False, temperature=0.6)
    if reply is None:
        return "Kërkohet çelësi Gemini (te 'Secrets' në Streamlit) që Verifikimi AI të përgjigjet."
    return reply


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
# SVG IKONA
# ---------------------------------------------------------------------------
ICON_CHECK = """<svg width="22" height="22" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
<path d="M9 12.5L11 14.5L15.5 9.5" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
<circle cx="12" cy="12" r="9" stroke="currentColor" stroke-width="2"/></svg>"""

ICON_SHIELD = """<svg width="22" height="22" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
<path d="M12 3L19 6V11C19 15.5 16 19 12 21C8 19 5 15.5 5 11V6L12 3Z" stroke="currentColor" stroke-width="2" stroke-linejoin="round"/></svg>"""

ICON_PEOPLE = """<svg width="22" height="22" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
<circle cx="9" cy="8" r="3" stroke="currentColor" stroke-width="2"/>
<circle cx="17" cy="9" r="2.4" stroke="currentColor" stroke-width="2"/>
<path d="M3.5 19C3.5 15.5 6 13.5 9 13.5C12 13.5 14.5 15.5 14.5 19" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
<path d="M14.8 14.2C17.2 14.4 19 16.1 19.5 19" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>"""

ICON_BOLT = """<svg width="22" height="22" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
<path d="M13 3L5 13.5H11L10.5 21L19 9.5H13L13 3Z" stroke="currentColor" stroke-width="2" stroke-linejoin="round"/></svg>"""

ICON_CHAT = """<svg width="26" height="26" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
<path d="M4 12C4 7.58 7.58 4 12 4C16.42 4 20 7.58 20 12C20 16.42 16.42 20 12 20C10.6 20 9.28 19.64 8.13 19L4 20L5.13 16.35C4.42 15.13 4 13.62 4 12Z" stroke="currentColor" stroke-width="2" stroke-linejoin="round"/></svg>"""

ICON_ANALYSIS = """<svg width="26" height="26" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
<rect x="4" y="4" width="16" height="16" rx="3" stroke="currentColor" stroke-width="2"/>
<path d="M8 14L11 11L13 13L16 9" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>"""

ICON_SCALE = """<svg width="18" height="18" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
<path d="M12 3V21M6 6H18M6 6L3 12H9L6 6ZM18 6L15 12H21L18 6Z" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>"""


# ---------------------------------------------------------------------------
# UI SETUP - TEMA E ERRET, MINIMALE
# ---------------------------------------------------------------------------
st.set_page_config(page_title=APP_TITLE, page_icon="🛡️", layout="centered")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp { background: #0c0c10; }
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stToolbar"] { display: none; }
    .block-container { padding-top: 2rem; max-width: 720px; }

    h1, h2, h3, p, span, label, div { color: #ececf1; }

    .app-title { font-size: 1.6rem; font-weight: 800; letter-spacing: -0.4px; margin-bottom: 0.15rem; }
    .app-subtitle { font-size: 0.88rem; color: #8b8b96 !important; margin-bottom: 1.4rem; line-height: 1.4; }
    .headline-text { font-size: 1.3rem; font-weight: 800; line-height: 1.35; margin: 0.3rem 0 1.1rem 0; }

    .app-card {
        background: #16161c; border: 1px solid #222229; border-radius: 18px;
        padding: 1.2rem 1.3rem; margin-bottom: 1rem;
    }
    .app-card-title { font-weight: 700; font-size: 0.95rem; color: #ececf1 !important; margin-bottom: 0.7rem; }
    .section-label {
        font-weight: 700; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.04em;
        color: #6d6d78 !important; margin: 1.1rem 0 0.6rem 0;
    }

    div[role="radiogroup"] { gap: 0.5rem; }
    div[role="radiogroup"] label { color: #ececf1 !important; }

    .stButton > button {
        border-radius: 12px !important; font-weight: 600 !important; border: 1px solid #26262f !important;
        background: #1b1b22 !important; color: #c9c9d2 !important;
    }
    .stButton > button:hover { border-color: #6d5bd0 !important; color: #ffffff !important; }
    .stButton > button[kind="primary"] { background: #6d5bd0 !important; color: #ffffff !important; border: none !important; }
    .stButton > button[kind="primary"]:hover { background: #7c6bdb !important; color: #ffffff !important; }
    .stButton > button:disabled { background: #1b1b22 !important; color: #4b4b54 !important; border: 1px solid #222229 !important; }

    [data-baseweb="tab-list"] { gap: 0.3rem; background: transparent; border-bottom: 1px solid #222229; }
    [data-baseweb="tab"] { border-radius: 0 !important; font-weight: 600; color: #8b8b96 !important; }
    [aria-selected="true"][data-baseweb="tab"] { color: #ffffff !important; border-bottom: 2px solid #6d5bd0 !important; }

    textarea, input { background: #101014 !important; color: #ececf1 !important; border-color: #26262f !important; border-radius: 12px !important; }

    [data-testid="stExpander"] { background: #16161c !important; border: 1px solid #222229 !important; border-radius: 14px !important; }

    /* ---------- MENU FILLESTAR ---------- */
    .menu-card {
        background: #16161c; border: 1px solid #222229; border-radius: 20px;
        padding: 1.5rem 1.4rem; margin-bottom: 1rem; display: flex; gap: 1rem; align-items: center;
        cursor: pointer;
    }
    .menu-icon {
        flex-shrink: 0; width: 52px; height: 52px; border-radius: 14px; background: rgba(109,91,208,0.16);
        color: #a99bf0; display: flex; align-items: center; justify-content: center;
    }
    .menu-title { font-weight: 700; font-size: 1.05rem; margin-bottom: 0.2rem; }
    .menu-desc { font-size: 0.82rem; color: #8b8b96 !important; line-height: 1.4; }

    /* ---------- 4 GAUGE SCORE CARDS ---------- */
    .score-row { display: flex; gap: 0.6rem; margin: 0.2rem 0 1rem 0; }
    .score-card { flex: 1; background: #16161c; border: 1px solid #222229; border-radius: 16px; padding: 0.85rem 0.4rem; text-align: center; }
    .score-card-label { font-size: 0.72rem; color: #9b9ba6 !important; font-weight: 600; margin-bottom: 0.6rem; }
    .score-gauge { width: 60px; height: 60px; border-radius: 50%; margin: 0 auto; display: flex; align-items: center; justify-content: center; }
    .score-gauge-inner { width: 48px; height: 48px; border-radius: 50%; background: #16161c; display: flex; align-items: center; justify-content: center; }

    /* ---------- KEY FINDINGS / VERIFIED FACTS ---------- */
    .finding-card { background: #16161c; border: 1px solid #222229; border-radius: 14px; padding: 0.75rem 1rem; margin-bottom: 0.5rem; }
    .finding-tag {
        display: inline-block; font-size: 0.66rem; font-weight: 700; text-transform: uppercase;
        color: #a99bf0 !important; background: rgba(109,91,208,0.16); padding: 0.12rem 0.55rem;
        border-radius: 999px; margin-bottom: 0.35rem; letter-spacing: 0.03em;
    }
    .finding-text { color: #d0d0d8 !important; font-size: 0.88rem; line-height: 1.4; }
    .fact-row { display: flex; gap: 0.6rem; align-items: flex-start; padding: 0.55rem 0; border-bottom: 1px solid #1e1e26; }
    .fact-row:last-child { border-bottom: none; }
    .fact-check { flex-shrink: 0; color: #22c55e; margin-top: 0.1rem; }

    /* ---------- DEBATE ---------- */
    .debate-row { display: flex; gap: 0.7rem; margin-bottom: 0.7rem; }
    .debate-avatar { flex-shrink: 0; width: 34px; height: 34px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: 700; font-size: 0.85rem; color: #ffffff !important; }
    .debate-bubble { flex: 1; border-radius: 14px; padding: 0.7rem 0.9rem; border: 1px solid; }
    .debate-bubble.supportive { background: rgba(59,130,246,0.08); border-color: rgba(59,130,246,0.25); }
    .debate-bubble.critical { background: rgba(239,68,68,0.08); border-color: rgba(239,68,68,0.25); }
    .debate-author { font-weight: 700; font-size: 0.8rem; margin-bottom: 0.15rem; }
    .debate-bubble.supportive .debate-author { color: #93c5fd !important; }
    .debate-bubble.critical .debate-author { color: #fca5a5 !important; }
    .debate-text { color: #d0d0d8 !important; font-size: 0.86rem; line-height: 1.4; }

    /* ---------- VERDICT ---------- */
    .final-verdict-card { background: #16161c; border: 1px solid rgba(109,91,208,0.4); border-radius: 16px; padding: 1.1rem 1.2rem; margin-top: 0.3rem; }
    .final-verdict-title { font-weight: 700; font-size: 0.95rem; color: #a99bf0 !important; margin-bottom: 0.6rem; display: flex; align-items: center; gap: 0.4rem; }
    .final-verdict-text { color: #ececf1 !important; font-size: 0.9rem; line-height: 1.55; }

    .disclaimer { font-size: 0.75rem; color: #55555f !important; text-align: center; margin-top: 1.4rem; line-height: 1.4; }

    /* ---------- CHAT ---------- */
    [data-testid="stChatMessage"] { background: #16161c !important; border: 1px solid #222229 !important; border-radius: 14px !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


def gauge_color(pct: float, invert: bool = False) -> str:
    v = (100 - pct) if invert else pct
    if v >= 70:
        return "#22c55e"
    if v >= 40:
        return "#f59e0b"
    return "#ef4444"


def score_card(label: str, icon_svg: str, pct: float, invert: bool = False):
    color = gauge_color(pct, invert=invert)
    deg = max(0, min(100, pct)) * 3.6
    st.markdown(
        f"""
        <div class="score-card">
            <div class="score-card-label">{label}</div>
            <div class="score-gauge" style="background: conic-gradient({color} {deg}deg, #24242c 0deg);">
                <div class="score-gauge-inner" style="color:{color};">{icon_svg}</div>
            </div>
            <div style="margin-top:0.4rem; font-weight:800; font-size:0.85rem; color:{color};">{pct:.0f}%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# NAVIGIMI - MENU FILLESTAR
# ---------------------------------------------------------------------------
if "page" not in st.session_state:
    st.session_state["page"] = "menu"


def go_to(page_name):
    st.session_state["page"] = page_name
    st.rerun()


def top_bar(current_title: str):
    c1, c2 = st.columns([1, 6])
    with c1:
        if st.button("←", key="back_btn"):
            go_to("menu")
    with c2:
        st.markdown(f'<div class="app-title" style="margin-top:0.3rem;">{current_title}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# FAQJA: MENU
# ---------------------------------------------------------------------------
if st.session_state["page"] == "menu":
    st.markdown(
        f"""
        <div class="app-title">{APP_TITLE}</div>
        <div class="app-subtitle">Zbulues i lajmeve të rreme në shqip — prototip praktik i punimit të diplomës
        "Evaluating Bias and Fairness of Multilingual Models on Albanian Fake News".</div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""<div class="menu-card"><div class="menu-icon">{ICON_ANALYSIS}</div>
        <div><div class="menu-title">Analizë e Detajuar</div>
        <div class="menu-desc">Ngjit një lajm ose link dhe merr Truth Score, Besueshmëri, Konsensus, Ndikim, gjetje kryesore dhe verdikt final.</div>
        </div></div>""",
        unsafe_allow_html=True,
    )
    if st.button("Hap Analizën", use_container_width=True, type="primary", key="open_analysis"):
        go_to("analysis")

    st.markdown(
        f"""<div class="menu-card"><div class="menu-icon">{ICON_CHAT}</div>
        <div><div class="menu-title">Verifikimi AI</div>
        <div class="menu-desc">Bisedë e lirë me AI — pyet për çdo lajm ose pretendim dhe merr përgjigje direkt.</div>
        </div></div>""",
        unsafe_allow_html=True,
    )
    if st.button("Hap Verifikimin AI", use_container_width=True, key="open_chat"):
        go_to("chat")

    if st.button("Gjetjet e Kërkimit (Diplomë)", use_container_width=True, key="open_findings"):
        go_to("findings")

    st.markdown(
        '<p class="disclaimer">Prototip akademik për demonstrim, jo mjet i verifikuar për prodhim.</p>',
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# FAQJA: ANALIZA E DETAJUAR
# ---------------------------------------------------------------------------
elif st.session_state["page"] == "analysis":
    top_bar("Analizë e Detajuar")

    st.markdown('<div class="app-card">', unsafe_allow_html=True)
    st.markdown('<div class="app-card-title">Modeli</div>', unsafe_allow_html=True)
    model_choice = st.radio("Zgjidh modelin", list(MODELS.keys()), horizontal=True, label_visibility="collapsed")
    st.caption(f"{MODELS[model_choice]['description']} · saktësi few-shot: {MODELS[model_choice]['accuracy']}")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="app-card">', unsafe_allow_html=True)
    st.markdown('<div class="app-card-title">Vendos titullin ose lajmin</div>', unsafe_allow_html=True)

    ex_cols = st.columns(len(EXAMPLES))
    for i, (ex_name, ex_text) in enumerate(EXAMPLES.items()):
        if ex_cols[i].button(ex_name, use_container_width=True):
            st.session_state["input_text"] = ex_text
            st.session_state["fetched_text"] = ex_text

    input_mode = st.radio("Si do ta japësh lajmin?", ["Ngjit tekstin", "Vendos link (URL)"], horizontal=True, label_visibility="collapsed")

    if input_mode == "Ngjit tekstin":
        text = st.text_area(
            "Ngjit tekstin e një lajmi në shqip:",
            value=st.session_state.get("input_text", ""),
            height=160,
            placeholder="Ngjit titullin dhe/ose përmbajtjen e lajmit...",
            key="text_input_area",
            label_visibility="collapsed",
        )
        source_url = None
    else:
        source_url = st.text_input("Vendos linkun e artikullit:", placeholder="https://...", label_visibility="collapsed")
        if st.button("Merr artikullin"):
            if not source_url.strip():
                st.warning("Fut një link para se të vazhdosh.")
            else:
                try:
                    with st.spinner("Duke shkarkuar..."):
                        fetched = fetch_article_text(source_url.strip())
                    st.session_state["fetched_text"] = fetched
                    st.success(f"U morën {len(fetched)} karaktere.")
                except ValueError as e:
                    st.error(str(e))
        text = st.text_area("Teksti i marrë (mund ta redaktosh):", value=st.session_state.get("fetched_text", ""), height=160)

    analyze = st.button("ANALIZO", type="primary", use_container_width=True, disabled=not text.strip())
    st.markdown('</div>', unsafe_allow_html=True)

    if analyze:
        model_path = MODELS[model_choice]["path"]
        try:
            label, confidence, all_probs = predict(text, model_path)
            truth_pct = all_probs[0] * 100

            with st.spinner("Duke analizuar me AI..."):
                ruling = gemini_ruling(text, label, confidence)

            st.session_state["last_result"] = {
                "ruling": ruling, "label": label, "confidence": confidence,
                "truth_pct": truth_pct, "model_choice": model_choice,
                "source_url": source_url, "text": text,
            }
        except OSError:
            st.error(f"S'u gjet modeli te `{model_path}`. Kontrollo variablën MODELS.")

    result = st.session_state.get("last_result")
    if result:
        ruling = result["ruling"]
        label = result["label"]
        confidence = result["confidence"]
        truth_pct = result["truth_pct"]
        model_choice = result["model_choice"]

        st.markdown(f'<div class="headline-text">{ruling.get("headline", "")}</div>', unsafe_allow_html=True)

        result_tab_analysis, result_tab_verdict = st.tabs(["Analiza", "Verdikti"])

        with result_tab_analysis:
            st.markdown('<div class="section-label">Rezultatet</div>', unsafe_allow_html=True)
            st.markdown('<div class="score-row">', unsafe_allow_html=True)
            cols = st.columns(4)
            with cols[0]:
                score_card("Truth Score", ICON_CHECK, truth_pct)
            with cols[1]:
                score_card("Besueshmëri", ICON_SHIELD, float(ruling.get("reliability_score", 50)))
            with cols[2]:
                score_card("Konsensus", ICON_PEOPLE, float(ruling.get("consensus_score", 50)))
            with cols[3]:
                score_card("Ndikim", ICON_BOLT, float(ruling.get("impact_score", 50)), invert=True)
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="app-card">', unsafe_allow_html=True)
            st.markdown('<div class="app-card-title">Përmbledhje</div>', unsafe_allow_html=True)
            st.markdown(f'<p class="finding-text">{ruling.get("analysis_summary", "")}</p>', unsafe_allow_html=True)
            st.caption(
                f"Modeli i përdorur: {model_choice} ({MODELS[model_choice]['description']}) · "
                f"saktësi few-shot {MODELS[model_choice]['accuracy']} · klasifikim: {label} ({confidence*100:.1f}%)"
            )
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="section-label">Gjetjet Kryesore</div>', unsafe_allow_html=True)
            for finding in ruling.get("key_findings", []):
                st.markdown(
                    f"""<div class="finding-card">
                        <span class="finding-tag">{finding.get('tag','')}</span>
                        <div class="finding-text">{finding.get('text','')}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )

            st.markdown('<div class="section-label">Debati</div>', unsafe_allow_html=True)
            sup = ruling.get("debate_supportive", {})
            crit = ruling.get("debate_critical", {})
            sup_initial = (sup.get("author") or "?")[:1].upper()
            crit_initial = (crit.get("author") or "?")[:1].upper()
            st.markdown(
                f"""<div class="debate-row">
                    <div class="debate-avatar" style="background:#3b82f6;">{sup_initial}</div>
                    <div class="debate-bubble supportive">
                        <div class="debate-author">{sup.get('author','')}</div>
                        <div class="debate-text">{sup.get('text','')}</div>
                    </div>
                </div>
                <div class="debate-row">
                    <div class="debate-avatar" style="background:#ef4444;">{crit_initial}</div>
                    <div class="debate-bubble critical">
                        <div class="debate-author">{crit.get('author','')}</div>
                        <div class="debate-text">{crit.get('text','')}</div>
                    </div>
                </div>""",
                unsafe_allow_html=True,
            )

            if result.get("source_url"):
                st.markdown('<div class="section-label">Burimi</div>', unsafe_allow_html=True)
                st.markdown(f"[{result['source_url']}]({result['source_url']})")

        with result_tab_verdict:
            with st.expander("Pjesëmarrësit"):
                st.markdown(f"**{sup.get('author','')}** — argument mbështetës")
                st.markdown(f"**{crit.get('author','')}** — argument kritik")
                st.markdown(f"**{model_choice}** — modeli klasifikues ({MODELS[model_choice]['accuracy']} saktësi)")

            with st.expander("Të Dhënat e Rastit"):
                st.text(result["text"][:1500])

            verdict_label = "LAJM I RREMË" if label == "FAKE" else "LAJM I BESUESHËM"
            verdict_color = "#ef4444" if label == "FAKE" else "#22c55e"
            st.markdown(
                f"""
                <div class="final-verdict-card">
                    <div class="final-verdict-title" style="color:{verdict_color} !important;">
                        {ICON_SCALE} {verdict_label} · Truth Score {truth_pct:.0f}%
                    </div>
                    <div class="final-verdict-text">{ruling.get("verdict", "")}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown('<div class="section-label">Fakte të Verifikuara</div>', unsafe_allow_html=True)
            for finding in ruling.get("key_findings", []):
                st.markdown(
                    f"""<div class="fact-row"><span class="fact-check">{ICON_CHECK}</span>
                    <div class="finding-text">{finding.get('text','')}</div></div>""",
                    unsafe_allow_html=True,
                )

    st.markdown(
        '<p class="disclaimer">Prototip akademik për demonstrim, jo mjet i verifikuar për prodhim. '
        'Rezultatet (përfshi ato të gjeneruara nga AI) duhen interpretuar me kujdes.</p>',
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# FAQJA: VERIFIKIMI AI (CHAT)
# ---------------------------------------------------------------------------
elif st.session_state["page"] == "chat":
    top_bar("Verifikimi AI")
    st.markdown(
        '<p class="app-subtitle">Vendos një lajm ose pyetje më poshtë — Verifikimi AI përgjigjet direkt.</p>',
        unsafe_allow_html=True,
    )

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    for turn in st.session_state["chat_history"]:
        with st.chat_message("user" if turn["role"] == "user" else "assistant"):
            st.write(turn["text"])

    user_msg = st.chat_input("Vendos lajmin ose pyetjen tënde...")
    if user_msg:
        st.session_state["chat_history"].append({"role": "user", "text": user_msg})
        with st.chat_message("user"):
            st.write(user_msg)
        with st.chat_message("assistant"):
            with st.spinner("Duke menduar..."):
                reply = gemini_chat_reply(st.session_state["chat_history"])
            st.write(reply)
        st.session_state["chat_history"].append({"role": "assistant", "text": reply})

# ---------------------------------------------------------------------------
# FAQJA: GJETJET E KERKIMIT (permbajtje akademike)
# ---------------------------------------------------------------------------
elif st.session_state["page"] == "findings":
    top_bar("Gjetjet e Kërkimit")

    st.markdown('<div class="app-card">', unsafe_allow_html=True)
    st.markdown('<div class="app-card-title">Përmbledhje e eksperimenteve</div>', unsafe_allow_html=True)
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
    st.markdown('<div class="app-card-title">Bias sipas gjatësisë së artikullit</div>', unsafe_allow_html=True)
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
    st.markdown('<div class="app-card-title">Drejtësia ndër-gjuhësore (zero-shot)</div>', unsafe_allow_html=True)
    st.markdown(
        "XLM-R duket \"më i drejtë\" sipas Equal Opportunity Gap (0.003), por kjo është "
        "artificiale — modeli kishte kolapsuar duke parashikuar \"fake\" për çdo artikull. "
        "mBERT (EOG 0.212) dhe mT5 (EOG 0.488) pasqyrojnë hendekë realë EN→AL."
    )
    st.info("Për detaje të plota metodologjike, shih Kreun III–IV të punimit të diplomës.")
    st.markdown('</div>', unsafe_allow_html=True)
