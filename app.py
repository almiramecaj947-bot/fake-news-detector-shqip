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
 
import requests
import streamlit as st
import streamlit.components.v1 as components
import torch
import trafilatura
from transformers import AutoTokenizer, AutoModelForSequenceClassification
 
APP_TITLE = "TruthNews AL"
 
# ---------------------------------------------------------------------------
# PWA HEAD TAGS
# ---------------------------------------------------------------------------
# Streamlit Community Cloud serves its own prebuilt static/index.html from a
# CDN-fronted bundle (the "/-/build/..." assets) -- editing the installed
# streamlit package's index.html at runtime never reaches that served page,
# so Safari always fell back to Streamlit's own default icon/manifest.
# This injects the tags client-side instead, straight into the real page
# <head> via window.parent, which works regardless of how the HTML shell
# itself was served.
def _install_pwa_head_tags():
    manifest_url = (
        "https://raw.githubusercontent.com/almiramecaj947-bot/"
        "fake-news-detector-shqip/main/static/manifest.json"
    )
    icon_192 = (
        "https://raw.githubusercontent.com/almiramecaj947-bot/"
        "fake-news-detector-shqip/main/static/icon-192.png"
    )
    icon_512 = (
        "https://raw.githubusercontent.com/almiramecaj947-bot/"
        "fake-news-detector-shqip/main/static/icon-512.png"
    )
    components.html(
        f"""
        <script>
        (function() {{
            var doc = window.parent.document;
 
            function setLink(rel, href, sizes) {{
                var sel = 'link[rel="' + rel + '"]' + (sizes ? '[sizes="' + sizes + '"]' : '');
                doc.head.querySelectorAll(sel).forEach(function(el) {{ el.remove(); }});
                var link = doc.createElement('link');
                link.setAttribute('rel', rel);
                if (sizes) link.setAttribute('sizes', sizes);
                link.setAttribute('href', href);
                doc.head.appendChild(link);
            }}
            function setMeta(name, content) {{
                doc.head.querySelectorAll('meta[name="' + name + '"]').forEach(function(el) {{ el.remove(); }});
                var m = doc.createElement('meta');
                m.setAttribute('name', name);
                m.setAttribute('content', content);
                doc.head.appendChild(m);
            }}
 
            setLink('manifest', '{manifest_url}');
            setLink('apple-touch-icon', '{icon_192}');
            setLink('icon', '{icon_512}');
            setMeta('theme-color', '#0c1712');
            setMeta('apple-mobile-web-app-capable', 'yes');
            setMeta('apple-mobile-web-app-status-bar-style', 'black-translucent');
            setMeta('apple-mobile-web-app-title', 'TruthNews AL');
        }})();
        </script>
        """,
        height=0,
        width=0,
    )
 
 
# NOTE: called after st.set_page_config() below -- components.html() is a
# Streamlit command, and set_page_config() must be the very first Streamlit
# command run in the script or Streamlit raises an error.
 
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
 
# ---------------------------------------------------------------------------
# KONTEKST FAIRNESS-I (rezultate reale nga auditi i Kreut 4.7 te diploma)
# ---------------------------------------------------------------------------
# I RËNDËSISHËM: këto janë statistika NDËR-ARTIKUJSH (mbi 1.192 artikuj testimi),
# jo një "% bias" i vetë artikullit — Equal Opportunity Gap/FPR/FNR llogariten
# vetëm mbi grupe, jo mbi 1 rast të vetëm. Këtu përdoren si KONTEKST: në cilin
# grup bie artikulli i ri, dhe si ka performuar modeli historikisht për atë grup.
FAIRNESS_OVERALL = {"fpr": 1.17, "fnr": 4.38, "n": 1192}
 
TOPIC_KEYWORDS = {
    "Politikë": ["qeveri", "kryeministr", "parlament", "deputet", "zgjedhje", "president",
                 "opozit", "ministri", "kuvend", "partia", "kryetar bashkie"],
    "Ekonomi": ["ekonomi", "buxhet", "taksë", "inflacion", "lek", "euro", "bankë", "biznes",
                "investim", "papunësi", "pagë"],
    "Shëndetësi": ["spital", "covid", "virus", "vaksin", "mjek", "sëmundje", "pandemi",
                   "shëndetësor", "infeksion"],
    "Krim/Siguri": ["policia", "vrasje", "arrestim", "krim", "aksident", "gjykata", "hetim",
                    "drogë", "terrorist"],
    "Sport/Argëtim": ["futboll", "kampion", "ndeshje", "aktor", "këngëtar", "koncert", "film",
                      "muzikë", "sport"],
}
 
# Nga run/data/rezultate_tema.csv (12 shtator 2026, pool auditimi n=1.192)
FAIRNESS_TOPIC = {
    "Ekonomi":       {"n": 99,  "recall": 98.11, "fpr": 0.00, "fnr": 1.89,  "flag": False},
    "Krim/Siguri":   {"n": 112, "recall": 100.0,  "fpr": 0.00, "fnr": 0.00,  "flag": False},
    "Politikë":      {"n": 391, "recall": 94.33, "fpr": 1.02, "fnr": 5.67,  "flag": False},
    "Shëndetësi":    {"n": 264, "recall": 91.35, "fpr": 1.88, "fnr": 8.65,  "flag": True},
    "Sport/Argëtim": {"n": 119, "recall": 93.75, "fpr": 0.00, "fnr": 6.25,  "flag": False},
    "Tjetër":        {"n": 207, "recall": 98.47, "fpr": 2.63, "fnr": 1.53,  "flag": False},
}
 
# Nga run/data/rezultate_stili.csv — kufijtë (13, 24) janë tertilet e numrit të
# fjalëve TËRËSISHT me shkronja të mëdha, mbi po atë pool auditimi
STYLE_BOUNDS = (13, 24)
FAIRNESS_STYLE = {
    "Neutral":        {"n": 398, "recall": 96.07, "fpr": 1.78, "fnr": 3.93, "flag": False},
    "Mesatar":        {"n": 397, "recall": 96.67, "fpr": 0.64, "fnr": 3.33, "flag": False},
    "Sensacionalist": {"n": 397, "recall": 92.80, "fpr": 1.10, "fnr": 7.20, "flag": True},
}
 
# Nga run/data/rezultate_burimi_domain.csv — VETËM accuracy (jo FPR/FNR: shih
# kufizimin metodologjik te 4.7.3 e diplomës, domain-i është pothuajse vetë etiketa)
FAIRNESS_DOMAIN = {
    "lifestoriesz.com": 80.00, "lajmealb.xyz": 84.78, "showbiziks.com": 85.71,
    "valetal.live": 87.50, "bit.ly": 88.30, "infokosova.net": 96.85,
    "gazetaexpress.com": 97.22, "mesazhi.com": 98.11, "promakale.com": 98.51,
    "botasot.info": 98.72, "kallxo.com": 100.0, "gazetaktuale.press": 100.0,
    "koha.net": 100.0, "kosova-sot.info": 100.0, "bdnewsfeed24.com": 100.0,
    "audipassionmagazine.club": 100.0, "media24newss.com": 100.0,
    "mediaworldd.co": 100.0, "thedailybarta24.com": 100.0, "telegrafi.com": 100.0,
    "trendi-kspro.com": 100.0, "valetal.info": 100.0, "winingal.com": 100.0,
    "zeriamerikes.com": 100.0,
}
 
 
def _norm(t: str) -> str:
    return re.sub(r"\s+", " ", str(t)).strip().lower()
 
 
def detect_topic(text: str) -> str:
    tl = _norm(text)
    scores = {k: sum(tl.count(w) for w in ws) for k, ws in TOPIC_KEYWORDS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "Tjetër"
 
 
def detect_style(text: str) -> str:
    upper_count = sum(1 for w in str(text).split() if w.isupper() and len(w) > 1)
    low, high = STYLE_BOUNDS
    if upper_count <= low:
        return "Neutral"
    if upper_count <= high:
        return "Mesatar"
    return "Sensacionalist"
 
 
def detect_domain(source_url: str):
    if not source_url:
        return None
    try:
        from urllib.parse import urlparse
        netloc = urlparse(source_url if "://" in source_url else "https://" + source_url).netloc.lower()
        netloc = netloc.replace("www.", "")
        return netloc or None
    except Exception:
        return None
 
 
def fairness_context(text: str, source_url: str = None) -> dict:
    """Kontekst fairness-i për artikullin: në cilin grup bie (temë/stil/burim) dhe
    si ka performuar modeli historikisht për atë grup, sipas auditit të Kreut 4.7."""
    topic = detect_topic(text)
    style = detect_style(text)
    domain = detect_domain(source_url)
    return {
        "topic": topic, "topic_stats": FAIRNESS_TOPIC[topic],
        "style": style, "style_stats": FAIRNESS_STYLE[style],
        "domain": domain, "domain_accuracy": FAIRNESS_DOMAIN.get(domain) if domain else None,
    }
 
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
 
 
@st.cache_data(show_spinner=False)
def _chart_data_uri(rel_path: str) -> str:
    """Lexon një grafik PNG statik dhe e kthen si data-URI, që të futet direkt brenda
    një kartele HTML (st.image nuk mund të vendoset brenda një div-i të stilizuar,
    sepse Streamlit e nxjerr në një kontejner tjetër)."""
    import base64
    path = pathlib.Path(__file__).parent / rel_path
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"
 
 
def chart_card(rel_path: str, caption_html: str):
    try:
        uri = _chart_data_uri(rel_path)
    except FileNotFoundError:
        # Grafiku PNG mungon te repo-ja e vendosur (p.sh. u ngarkua vetëm app.py, jo
        # dhe static/charts/) — shfaqim një njoftim të qetë në vend që të prishim
        # gjithë faqen me një error.
        st.markdown(
            f"""<div class="chart-card">
                <div class="fair-note">⚠ Grafiku "{rel_path}" nuk u gjet te repo-ja — ngarko dosjen static/charts/ te GitHub.</div>
                <div class="chart-caption">{caption_html}</div>
            </div>""",
            unsafe_allow_html=True,
        )
        return
    st.markdown(
        f"""<div class="chart-card">
            <img src="{uri}" alt="" />
            <div class="chart-caption">{caption_html}</div>
        </div>""",
        unsafe_allow_html=True,
    )
 
 
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
 
ICON_SCALE_LG = """<svg width="26" height="26" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
<path d="M12 3V21M6 6H18M6 6L3 12H9L6 6ZM18 6L15 12H21L18 6Z" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>"""
 
ICON_MENU = """<svg width="22" height="22" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
<path d="M4 6H20M4 12H20M4 18H20" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"/></svg>"""
 
ICON_USER = """<svg width="22" height="22" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
<circle cx="12" cy="8" r="3.6" stroke="currentColor" stroke-width="2"/>
<path d="M4.5 20C4.5 15.9 7.85 13 12 13C16.15 13 19.5 15.9 19.5 20" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>"""
 
ICON_GAUGE = """<svg width="22" height="22" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
<path d="M4 17C4 11.48 7.58 7 12 7C16.42 7 20 11.48 20 17" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
<path d="M12 17L15.5 11.5" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
<circle cx="12" cy="17" r="1.3" fill="currentColor"/></svg>"""
 
ICON_INFO = """<svg width="22" height="22" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
<circle cx="12" cy="12" r="9" stroke="currentColor" stroke-width="2"/>
<path d="M12 11V16.5" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
<circle cx="12" cy="8" r="1.1" fill="currentColor"/></svg>"""
 
 
# ---------------------------------------------------------------------------
# UI SETUP - TEMA E ÇELËT, SI NË MOCKUP
# ---------------------------------------------------------------------------
st.set_page_config(page_title=APP_TITLE, page_icon="🛡️", layout="centered")
_install_pwa_head_tags()
 
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
 
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp { background: #f2f2ef; }
    /* Shënim: st.columns() e Streamlit-it i "palos" kolonat vertikalisht nën ~640px gjerësi —
       gjë që del keq për rreshta me shumë elementë në gjerësi telefoni. Për rreshtat
       vetëm-për-shfaqje (4 kutitë e score-it, bias/fairness) ndërtohen si NJË div flex
       me HTML (shih render_score_row/bias_fairness_row) në vend të st.columns, në mënyrë
       që të mbeten gjithmonë horizontale pavarësisht gjerësisë. st.columns përdoret vetëm
       kur na duhen widget-e reale (butona/radio), ku sjellja e Streamlit-it (stakim vertikal
       në telefon) është e pranueshme. */
    /* Shiriti sipër: ikonat e hamburger-it/profilit dalin nga rrjedha normale dhe
       "ngjiten" në cepat e block-container-it, që titulli mund të qëndrojë i qendërzuar
       pa u prekur nga sjellja e paqëndrueshme e st.columns në gjerësi telefoni. */
    .block-container { position: relative; }
    div[class*="st-key-iconbtn_menu"] { position: absolute; top: 0.15rem; left: 0; z-index: 999; }
    div[class*="st-key-iconbtn_profile"] { position: absolute; top: 0.15rem; right: 0; z-index: 999; }
    div[class*="st-key-iconbtn_"] [data-testid="stElementContainer"] { margin: 0 !important; }
    div[class*="st-key-iconbtn_"] [data-testid="stVerticalBlockBorderWrapper"] { margin: 0 !important; }
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    [data-testid="stHeader"] { background: transparent; pointer-events: none; }
    [data-testid="stToolbar"] { display: none; }
    .block-container { padding-top: 1.1rem; max-width: 720px; }
 
    h1, h2, h3, p, span, label, div { color: #17171a; }
 
    /* ---------- TOP BAR ---------- */
    .topbar-row { display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.4rem; }
    .app-title { font-size: 1.5rem; font-weight: 800; letter-spacing: -0.4px; margin: 0; color: #17171a !important; }
    .app-title .accent { color: #1fa971 !important; }
    div[class*="st-key-iconbtn_"] button {
        border-radius: 999px !important; width: 42px !important; height: 42px !important; padding: 0 !important;
        background: #ffffff !important; border: 1px solid #e4e4de !important; color: #4a4a52 !important;
        box-shadow: 0 2px 8px -4px rgba(0,0,0,0.15) !important; display: flex; align-items: center; justify-content: center;
    }
    div[class*="st-key-iconbtn_"] button:hover { border-color: #1fa971 !important; color: #1fa971 !important; }
    div[class*="st-key-iconbtn_"] button p { font-size: 1.1rem !important; }
 
    .app-eyebrow {
        display: inline-flex; align-items: center; gap: 0.35rem; font-size: 0.72rem; font-weight: 700;
        text-transform: uppercase; letter-spacing: 0.06em; color: #1fa971 !important;
        background: rgba(31,169,113,0.12); border: 1px solid rgba(31,169,113,0.3);
        padding: 0.22rem 0.65rem; border-radius: 999px; margin-bottom: 0.6rem;
    }
    .app-subtitle { font-size: 0.88rem; color: #6b6b76 !important; margin-bottom: 1.2rem; line-height: 1.45; }
    .headline-text { font-size: 1.25rem; font-weight: 800; line-height: 1.35; margin: 0.3rem 0 1.1rem 0; color: #17171a !important; }
 
    .app-card {
        background: #ffffff; border: 1px solid #ebebe6; border-radius: 18px;
        padding: 1.2rem 1.3rem; margin-bottom: 1rem;
        box-shadow: 0 10px 24px -16px rgba(20,20,15,0.18);
    }
    /* Streamlit st.container(border=True, key="card_...") wraps real widgets (radio,
       text_area, buttons) that can't live inside a raw markdown div — st-key-card_*
       is Streamlit's own stable class for that container, restyled to match .app-card */
    div[class*="st-key-card_"] {
        background: #ffffff !important; border: 1px solid #ebebe6 !important;
        border-radius: 18px !important; padding: 1.2rem 1.3rem !important;
        box-shadow: 0 10px 24px -16px rgba(20,20,15,0.18);
    }
    .app-card-title { font-weight: 700; font-size: 0.95rem; color: #17171a !important; margin-bottom: 0.7rem; }
    .section-label {
        font-weight: 700; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.04em;
        color: #8a8a92 !important; margin: 1.1rem 0 0.6rem 0;
    }
 
    div[role="radiogroup"] { gap: 0.5rem; }
    div[role="radiogroup"] label { color: #17171a !important; }
 
    .stButton > button {
        border-radius: 12px !important; font-weight: 600 !important; border: 1px solid #e4e4de !important;
        background: #ffffff !important; color: #3a3a42 !important;
    }
    .stButton > button:hover { border-color: #1fa971 !important; color: #148a5c !important; }
    .stButton > button[kind="primary"] { background: #1fa971 !important; color: #ffffff !important; border: none !important; }
    .stButton > button[kind="primary"]:hover { background: #17925f !important; color: #ffffff !important; }
    .stButton > button:disabled { background: #f2f2ef !important; color: #b6b6ae !important; border: 1px solid #e4e4de !important; }
    /* Streamlit fokuson/kliminon butonat me një unazë të kuqe-zezë të parazgjedhur —
       e mbajmë gjithmonë jeshile, si ngjyra e markës. */
    .stButton > button:focus, .stButton > button:focus-visible, .stButton > button:active {
        border-color: #1fa971 !important; color: #148a5c !important;
        box-shadow: 0 0 0 2px rgba(31,169,113,0.25) !important; outline: none !important;
    }
    .stButton > button[kind="primary"]:focus, .stButton > button[kind="primary"]:active {
        background: #17925f !important; color: #ffffff !important;
        box-shadow: 0 0 0 2px rgba(31,169,113,0.35) !important;
    }
    div[role="radiogroup"] label:focus-within, div[role="radiogroup"] label:has(input:checked) {
        color: #148a5c !important;
    }
    button:focus-visible { outline-color: #1fa971 !important; }
 
    [data-baseweb="tab-list"] { gap: 0.3rem; background: transparent; border-bottom: 1px solid #ebebe6; }
    [data-baseweb="tab"] { border-radius: 0 !important; font-weight: 600; color: #8a8a92 !important; }
    [aria-selected="true"][data-baseweb="tab"] { color: #17171a !important; border-bottom: 2px solid #1fa971 !important; }
 
    textarea, input { background: #fbfbf9 !important; color: #17171a !important; border-color: #e4e4de !important; border-radius: 12px !important; }
 
    [data-testid="stExpander"] { background: #ffffff !important; border: 1px solid #ebebe6 !important; border-radius: 14px !important; }
    [data-testid="stPopoverBody"] { background: #ffffff !important; border-radius: 16px !important; }
 
    /* ---------- MENU FILLESTAR ---------- */
    .hero-card {
        background: linear-gradient(135deg, #1fa971 0%, #17925f 100%); border-radius: 22px;
        padding: 1.7rem 1.5rem; margin-bottom: 1.1rem; box-shadow: 0 14px 30px -16px rgba(31,169,113,0.55);
        color: #ffffff !important;
    }
    .hero-card * { color: #ffffff !important; }
    .hero-icon { width: 46px; height: 46px; border-radius: 13px; background: rgba(255,255,255,0.2); display: flex; align-items: center; justify-content: center; margin-bottom: 0.7rem; }
    .hero-title { font-weight: 800; font-size: 1.15rem; margin-bottom: 0.2rem; }
    .hero-desc { font-size: 0.84rem; opacity: 0.92; line-height: 1.4; }
    div[class*="st-key-hero_btn"] button {
        background: #ffffff !important; color: #148a5c !important; font-weight: 700 !important; border: none !important;
        margin-top: 0.9rem !important;
    }
    div[class*="st-key-hero_btn"] button:hover { background: #f2f2ef !important; color: #0f6f49 !important; }
 
    .menu-card {
        background: #ffffff; border: 1px solid #ebebe6; border-radius: 18px;
        padding: 1.1rem 1.2rem; margin-bottom: 0.7rem; display: flex; gap: 0.9rem; align-items: center;
        box-shadow: 0 8px 20px -16px rgba(20,20,15,0.15);
    }
    .menu-icon {
        flex-shrink: 0; width: 44px; height: 44px; border-radius: 12px; background: rgba(42,120,214,0.12);
        color: #2a78d6; display: flex; align-items: center; justify-content: center;
    }
    .menu-icon.good { background: rgba(31,169,113,0.16); color: #1fa971; }
    .menu-title { font-weight: 700; font-size: 0.98rem; margin-bottom: 0.15rem; color: #17171a !important; }
    .menu-desc { font-size: 0.8rem; color: #6b6b76 !important; line-height: 1.35; }
    .menu-badge {
        display: inline-block; font-size: 0.6rem; font-weight: 700; text-transform: uppercase;
        letter-spacing: 0.03em; color: #1fa971 !important; background: rgba(31,169,113,0.14);
        padding: 0.08rem 0.5rem; border-radius: 999px; margin-left: 0.4rem; vertical-align: middle;
    }
 
    /* ---------- MENU/PROFILE DROPDOWN ---------- */
    div[class*="st-key-nav_"] button {
        justify-content: flex-start !important; background: transparent !important; border: none !important;
        padding: 0.5rem 0.2rem !important; font-weight: 600 !important; color: #3a3a42 !important;
    }
    div[class*="st-key-nav_"] button:hover { color: #1fa971 !important; }
 
    /* ---------- 4 GAUGE SCORE CARDS ---------- */
    .score-row { display: flex; gap: 0.55rem; margin: 0.2rem 0 1rem 0; }
    .score-card { flex: 1; background: #ffffff; border: 1px solid #ebebe6; border-radius: 16px; padding: 0.8rem 0.35rem; text-align: center; box-shadow: 0 8px 20px -16px rgba(20,20,15,0.15); }
    .score-card-label { font-size: 0.68rem; color: #6b6b76 !important; font-weight: 600; margin-bottom: 0.35rem; }
    .score-card-value { font-size: 1.05rem; font-weight: 800; margin-bottom: 0.5rem; }
    .score-gauge { width: 52px; height: 52px; border-radius: 50%; margin: 0 auto; display: flex; align-items: center; justify-content: center; }
    .score-gauge-inner { width: 42px; height: 42px; border-radius: 50%; background: #ffffff; display: flex; align-items: center; justify-content: center; }
 
    /* ---------- BIAS / FAIRNESS / METRIKA (3 KUTI) ---------- */
    .bf-row { display: flex; gap: 0.55rem; margin-bottom: 0.5rem; }
    .bf-card { flex: 1; background: #ffffff; border: 1px solid #ebebe6; border-radius: 16px; padding: 0.75rem 0.55rem; text-align: center; box-shadow: 0 8px 20px -16px rgba(20,20,15,0.15); }
    .bf-card-label { font-size: 0.64rem; color: #6b6b76 !important; font-weight: 700; text-transform: uppercase; letter-spacing: 0.02em; margin-bottom: 0.4rem; }
    .bf-card-value { font-size: 1.2rem; font-weight: 800; margin-bottom: 0.15rem; }
    .bf-card-sub { font-size: 0.66rem; color: #9a9aa0 !important; line-height: 1.3; }
    .bf-metrics-list { text-align: left; font-size: 0.72rem; color: #4a4a52 !important; line-height: 1.55; }
    .bf-metrics-list b { color: #17171a !important; }
 
    /* ---------- KEY FINDINGS / VERIFIED FACTS ---------- */
    .finding-card { background: #ffffff; border: 1px solid #ebebe6; border-radius: 14px; padding: 0.75rem 1rem; margin-bottom: 0.5rem; box-shadow: 0 6px 16px -14px rgba(20,20,15,0.15); }
    .finding-tag {
        display: inline-block; font-size: 0.64rem; font-weight: 700; text-transform: uppercase;
        color: #2a78d6 !important; background: rgba(42,120,214,0.12); padding: 0.12rem 0.55rem;
        border-radius: 999px; margin-bottom: 0.35rem; letter-spacing: 0.03em;
    }
    .finding-text { color: #2c2c33 !important; font-size: 0.87rem; line-height: 1.4; }
 
    /* ---------- KONTEKST FAIRNESS-I (detajet, poshtë kutive) ---------- */
    .fair-badge-row { display: flex; gap: 0.6rem; flex-wrap: wrap; margin-bottom: 0.5rem; }
    .fair-badge { flex: 1; min-width: 150px; background: #fbfbf9; border: 1px solid #ebebe6; border-radius: 14px; padding: 0.7rem 0.85rem; }
    .fair-badge.flagged { border-color: rgba(224,82,79,0.4); background: rgba(224,82,79,0.06); }
    .fair-badge-label { font-size: 0.66rem; text-transform: uppercase; letter-spacing: 0.03em; color: #8a8a92 !important; font-weight: 700; margin-bottom: 0.25rem; }
    .fair-badge-group { font-size: 0.93rem; font-weight: 700; color: #17171a !important; margin-bottom: 0.3rem; }
    .fair-badge-stat { font-size: 0.76rem; color: #4a4a52 !important; line-height: 1.35; }
    .fair-badge-flag { font-size: 0.7rem; color: #c23c3a !important; font-weight: 600; margin-top: 0.3rem; }
    .fair-note { font-size: 0.74rem; color: #8a8a92 !important; line-height: 1.5; margin-top: 0.6rem; }
    .fact-row { display: flex; gap: 0.6rem; align-items: flex-start; padding: 0.55rem 0; border-bottom: 1px solid #efefe9; }
    .fact-row:last-child { border-bottom: none; }
    .fact-check { flex-shrink: 0; color: #1fa971; margin-top: 0.1rem; }
 
    /* ---------- DEBATE ---------- */
    .debate-row { display: flex; gap: 0.7rem; margin-bottom: 0.7rem; }
    .debate-avatar { flex-shrink: 0; width: 34px; height: 34px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: 700; font-size: 0.85rem; color: #ffffff !important; }
    .debate-bubble { flex: 1; border-radius: 14px; padding: 0.7rem 0.9rem; border: 1px solid; }
    .debate-bubble.supportive { background: rgba(42,120,214,0.08); border-color: rgba(42,120,214,0.25); }
    .debate-bubble.critical { background: rgba(224,82,79,0.08); border-color: rgba(224,82,79,0.25); }
    .debate-author { font-weight: 700; font-size: 0.8rem; margin-bottom: 0.15rem; }
    .debate-bubble.supportive .debate-author { color: #2a78d6 !important; }
    .debate-bubble.critical .debate-author { color: #c23c3a !important; }
    .debate-text { color: #2c2c33 !important; font-size: 0.86rem; line-height: 1.4; }
 
    /* ---------- VERDICT ---------- */
    .final-verdict-card { background: #ffffff; border: 1px solid rgba(31,169,113,0.4); border-radius: 16px; padding: 1.1rem 1.2rem; margin-top: 0.3rem; box-shadow: 0 10px 24px -16px rgba(20,20,15,0.18); }
    .final-verdict-title { font-weight: 700; font-size: 0.95rem; color: #148a5c !important; margin-bottom: 0.6rem; display: flex; align-items: center; gap: 0.4rem; }
    .final-verdict-text { color: #17171a !important; font-size: 0.9rem; line-height: 1.55; }
 
    .disclaimer { font-size: 0.74rem; color: #a5a5ac !important; text-align: center; margin-top: 1.4rem; line-height: 1.4; }
 
    /* ---------- CHARTS (Drejtësia & Bias-i) ---------- */
    .chart-card {
        background: #ffffff; border: 1px solid #ebebe6; border-radius: 18px;
        padding: 1rem 1rem 0.4rem 1rem; margin-bottom: 1rem; box-shadow: 0 10px 24px -16px rgba(20,20,15,0.18);
    }
    .chart-card img { border-radius: 10px; width: 100%; }
    .chart-caption { font-size: 0.8rem; color: #6b6b76 !important; line-height: 1.45; padding: 0.6rem 0.2rem 0.9rem 0.2rem; }
    .stat-pill-row { display: flex; gap: 0.5rem; flex-wrap: wrap; margin-bottom: 0.9rem; }
    .stat-pill { background: #ffffff; border: 1px solid #ebebe6; border-radius: 12px; padding: 0.55rem 0.8rem; flex: 1; min-width: 130px; box-shadow: 0 6px 16px -14px rgba(20,20,15,0.15); }
    .stat-pill-value { font-size: 1.15rem; font-weight: 800; color: #17171a !important; }
    .stat-pill-label { font-size: 0.7rem; color: #6b6b76 !important; margin-top: 0.1rem; }
 
    /* ---------- CHAT ---------- */
    [data-testid="stChatMessage"] { background: #ffffff !important; border: 1px solid #ebebe6 !important; border-radius: 14px !important; }
    </style>
    """,
    unsafe_allow_html=True,
)
 
 
def gauge_color(pct: float, invert: bool = False) -> str:
    v = (100 - pct) if invert else pct
    if v >= 70:
        return "#1fa971"
    if v >= 40:
        return "#e2a63a"
    return "#e0524f"
 
 
def fair_badge(label: str, group: str, stats: dict, overall_fnr: float) -> str:
    flagged = stats.get("flag", False)
    cls = "fair-badge flagged" if flagged else "fair-badge"
    flag_html = (
        f'<div class="fair-badge-flag">⚠ FNR/FPR ≥1,5× mesatares dataset-it</div>' if flagged else ""
    )
    html = f"""<div class="{cls}">
        <div class="fair-badge-label">{label}</div>
        <div class="fair-badge-group">{group}</div>
        <div class="fair-badge-stat">Recall: {stats['recall']:.1f}% · FPR: {stats['fpr']:.1f}% · FNR: {stats['fnr']:.1f}%</div>
        <div class="fair-badge-stat">(n={stats['n']} në grupin e testuar · mesatarja e dataset-it: FNR {overall_fnr:.1f}%)</div>
        {flag_html}
    </div>"""
    # Shih shënimin te _score_card_html: rrafshohet për t'u siguruar që 2 karta të
    # bashkuara (fair_badge(...) + fair_badge(...)) nuk lënë rresht bosh mes tyre.
    return re.sub(r"\s+", " ", html).strip()
 
 
def _score_card_html(label: str, icon_svg: str, pct: float, invert: bool = False) -> str:
    color = gauge_color(pct, invert=invert)
    deg = max(0, min(100, pct)) * 3.6
    html = f"""<div class="score-card">
            <div class="score-card-label">{label}</div>
            <div class="score-card-value" style="color:{color};">{pct:.0f}%</div>
            <div class="score-gauge" style="background: conic-gradient({color} {deg}deg, #ebebe6 0deg);">
                <div class="score-gauge-inner" style="color:{color};">{icon_svg}</div>
            </div>
        </div>"""
    # Rrafshohet në 1 rresht: një rresht bosh mes 2 blloqesh HTML e ndërpret bllokun
    # raw-HTML te CommonMark dhe pjesa pas tij bie në kod (indentim >=4 hapësira) —
    # shihet konkretisht kur bashkohen disa karta në 1 thirrje të vetme st.markdown.
    return re.sub(r"\s+", " ", html).strip()
 
 
def render_score_row(items: list):
    """Rendon kutitë e score-it (Truth Score, Besueshmëri, Konsensus, Ndikim) si NJË
    div flex në një thirrje të vetme st.markdown — jo me st.columns, sepse Streamlit
    i palos kolonat vertikalisht në gjerësi telefoni (shih shënimin te CSS më sipër)."""
    cards_html = "".join(_score_card_html(*item) for item in items)
    st.markdown(f'<div class="score-row">{cards_html}</div>', unsafe_allow_html=True)
 
 
_TOPIC_AVG_RECALL = sum(v["recall"] for v in FAIRNESS_TOPIC.values()) / len(FAIRNESS_TOPIC)
 
 
def bias_fairness_row(fairness: dict):
    """3 kuti koncize: Bias, Fairness, Metrika të tjera — kontekst nga auditi i Kreut 4.7,
    jo bias/fairness i vetë këtij 1 artikulli (shih shënimin metodologjik më poshtë)."""
    topic_stats = fairness["topic_stats"]
    style_stats = fairness["style_stats"]
 
    bias_risk = max(topic_stats["fnr"], style_stats["fnr"])
    bias_color = gauge_color(bias_risk, invert=True)
 
    fairness_pct = max(0.0, min(100.0, 100 - abs(topic_stats["recall"] - _TOPIC_AVG_RECALL)))
    fairness_color = gauge_color(fairness_pct)
 
    st.markdown(
        f"""
        <div class="bf-row">
            <div class="bf-card">
                <div class="bf-card-label">Bias</div>
                <div class="bf-card-value" style="color:{bias_color};">{bias_risk:.1f}%</div>
                <div class="bf-card-sub">FNR e grupit "{fairness['topic']}"/"{fairness['style']}"</div>
            </div>
            <div class="bf-card">
                <div class="bf-card-label">Fairness</div>
                <div class="bf-card-value" style="color:{fairness_color};">{fairness_pct:.0f}%</div>
                <div class="bf-card-sub">Përputhje me mesataren e temave</div>
            </div>
            <div class="bf-card">
                <div class="bf-card-label">Metrika</div>
                <div class="bf-metrics-list">
                    <b>n</b> = {FAIRNESS_OVERALL['n']}<br/>
                    FPR dataset: <b>{FAIRNESS_OVERALL['fpr']:.1f}%</b><br/>
                    FNR dataset: <b>{FAIRNESS_OVERALL['fnr']:.1f}%</b>
                </div>
            </div>
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
 
 
NAV_ITEMS = [
    ("menu", "🏠  Faqja Kryesore"),
    ("analysis", "🔎  Analizo Lajm"),
    ("findings", "⚖️  Drejtësia & Bias-i"),
]
 
 
def top_bar(current_title: str = None):
    """Shiriti sipër: hamburger (majtas, hap menunë e navigimit), titulli i app-it i qendërzuar,
    dhe ikona e profilit (djathtas, hap një kartë të shkurtër 'Rreth'). Ikonat janë 'ngjitur'
    me CSS te cepat e block-container-it (shih .st-key-iconbtn_* më sipër) në vend që të
    përdorin st.columns, sepse kolonat e Streamlit-it sillen në mënyrë të paqëndrueshme
    në gjerësi telefoni (444px) kur detyrohen horizontale."""
    with st.popover("☰", use_container_width=False, key="iconbtn_menu"):
        for page_key, page_label in NAV_ITEMS:
            if st.button(page_label, key=f"nav_{page_key}", use_container_width=True):
                go_to(page_key)
    with st.popover(ICON_USER_TXT, use_container_width=False, key="iconbtn_profile"):
        st.markdown(f"**{APP_TITLE}**")
        st.caption(f"Autore: Almira Mecaj · Modele: {', '.join(MODELS.keys())}")
 
    title_html = f'<span class="accent">{APP_TITLE[-2:]}</span>'.join([APP_TITLE[:-2], ""]) if len(APP_TITLE) > 2 else APP_TITLE
    st.markdown(
        f'<div class="app-title" style="text-align:center; margin-top:0.35rem;">{title_html}</div>',
        unsafe_allow_html=True,
    )
    if current_title:
        st.markdown(f'<div class="section-label" style="margin-top:0.3rem;">{current_title}</div>', unsafe_allow_html=True)
 
 
ICON_USER_TXT = "👤"
 
 
# ---------------------------------------------------------------------------
# FAQJA: MENU
# ---------------------------------------------------------------------------
if st.session_state["page"] == "menu":
    top_bar()
 
    st.markdown(
        '<div class="app-subtitle" style="margin-top:0.3rem;">Zbulues i lajmeve të rreme në shqip — analizo çdo lajm për saktësi dhe anësi.</div>',
        unsafe_allow_html=True,
    )
 
    # ---------- KOMANDA KRYESORE: ANALIZO LAJM ----------
    st.markdown(
        f"""<div class="hero-card"><div class="hero-icon">{ICON_ANALYSIS}</div>
        <div class="hero-title">Analizo Lajm</div>
        <div class="hero-desc">Ngjit një lajm ose link dhe merr Truth Score, kontekst fairness-i sipas temës ose stilit, gjetje kryesore dhe verdikt final.</div>
        </div>""",
        unsafe_allow_html=True,
    )
    if st.button("Analizo Lajm →", use_container_width=True, type="primary", key="hero_btn_analysis"):
        go_to("analysis")
 
    st.markdown('<div class="section-label">Më Shumë</div>', unsafe_allow_html=True)
 
    st.markdown(
        f"""<div class="menu-card"><div class="menu-icon good">{ICON_SCALE_LG}</div>
        <div><div class="menu-title">Drejtësia &amp; Bias-i i Modelit<span class="menu-badge">Kërkimi</span></div>
        <div class="menu-desc">Rezultatet reale të vlerësimit të fairness-it nga diploma.</div>
        </div></div>""",
        unsafe_allow_html=True,
    )
    if st.button("Hap Drejtësinë & Bias-in", use_container_width=True, key="open_findings"):
        go_to("findings")
 
# ---------------------------------------------------------------------------
# FAQJA: ANALIZA E DETAJUAR
# ---------------------------------------------------------------------------
elif st.session_state["page"] == "analysis":
    top_bar("Analizë e Detajuar")
 
    with st.container(border=True, key="card_model"):
        st.markdown('<div class="app-card-title">Modeli</div>', unsafe_allow_html=True)
        model_choice = st.radio("Zgjidh modelin", list(MODELS.keys()), horizontal=True, label_visibility="collapsed")
        st.caption(f"{MODELS[model_choice]['description']} · saktësi few-shot: {MODELS[model_choice]['accuracy']}")
 
    with st.container(border=True, key="card_input"):
        st.markdown('<div class="app-card-title">Vendos titullin ose lajmin</div>', unsafe_allow_html=True)
 
     ex_cols = st.columns(len(EXAMPLES))
     for i, (ex_name, ex_text) in enumerate(EXAMPLES.items()):
         if ex_cols[i].button(ex_name, use_container_width=True):
             st.session_state["text_input_area"] = ex_text
             st.session_state["fetched_text"] = ex_text
 
        input_mode = st.radio("Si do ta japësh lajmin?", ["Ngjit tekstin", "Vendos link (URL)"], horizontal=True, label_visibility="collapsed")
 
        if input_mode == "Ngjit tekstin":
            text = st.text_area(
                "Ngjit tekstin e një lajmi në shqip:"
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
 
    if analyze:
        model_path = MODELS[model_choice]["path"]
        try:
            label, confidence, all_probs = predict(text, model_path)
            truth_pct = all_probs[0] * 100
            fairness = fairness_context(text, source_url)
 
            with st.spinner("Duke lexuar..."):
                ruling = gemini_ruling(text, label, confidence)
 
            st.session_state["last_result"] = {
                "ruling": ruling, "label": label, "confidence": confidence,
                "truth_pct": truth_pct, "model_choice": model_choice,
                "source_url": source_url, "text": text, "fairness": fairness,
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
            render_score_row([
                ("Truth Score", ICON_CHECK, truth_pct, False),
                ("Besueshmëri", ICON_SHIELD, float(ruling.get("reliability_score", 50)), False),
                ("Konsensus", ICON_PEOPLE, float(ruling.get("consensus_score", 50)), False),
                ("Ndikim", ICON_BOLT, float(ruling.get("impact_score", 50)), True),
            ])
 
            with st.container(border=True, key="card_summary"):
                st.markdown('<div class="app-card-title">Përmbledhje</div>', unsafe_allow_html=True)
                st.markdown(f'<p class="finding-text">{ruling.get("analysis_summary", "")}</p>', unsafe_allow_html=True)
                st.caption(
                    f"Modeli i përdorur: {model_choice} ({MODELS[model_choice]['description']}) · "
                    f"saktësi few-shot {MODELS[model_choice]['accuracy']} · klasifikim: {label} ({confidence*100:.1f}%)"
                )
 
            fairness = result.get("fairness")
            if fairness:
                with st.container(border=True, key="card_fairness"):
                    st.markdown('<div class="app-card-title">Bias &amp; Fairness (Kreu 4.7 i punimit)</div>', unsafe_allow_html=True)
                    bias_fairness_row(fairness)
                    st.markdown(
                        '<div class="fair-note">Kontekst nga vlerësimi i diplomës mbi 1.192 artikuj — jo bias/fairness i vetë '
                        'këtij 1 artikulli (EOG/FPR/FNR maten mbi grupe, jo mbi 1 rast).</div>',
                        unsafe_allow_html=True,
                    )
                    with st.expander("Detaje sipas grupit (temë/stil/burim)"):
                        badges_html = fair_badge(
                            "Tema e zbuluar", fairness["topic"], fairness["topic_stats"], FAIRNESS_OVERALL["fnr"]
                        ) + fair_badge(
                            "Stili i zbuluar", fairness["style"], fairness["style_stats"], FAIRNESS_OVERALL["fnr"]
                        )
                        st.markdown(f'<div class="fair-badge-row">{badges_html}</div>', unsafe_allow_html=True)
                        if fairness.get("domain") and fairness.get("domain_accuracy") is not None:
                            st.markdown(
                                f'<div class="fair-badge-stat">Burimi <b>{fairness["domain"]}</b> — '
                                f'accuracy historike: {fairness["domain_accuracy"]:.1f}% '
                                f'(shih Kreu 4.7.3; jo EOG, thjesht accuracy përshkrues).</div>',
                                unsafe_allow_html=True,
                            )
 
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
                    <div class="debate-avatar" style="background:#2a78d6;">{sup_initial}</div>
                    <div class="debate-bubble supportive">
                        <div class="debate-author">{sup.get('author','')}</div>
                        <div class="debate-text">{sup.get('text','')}</div>
                    </div>
                </div>
                <div class="debate-row">
                    <div class="debate-avatar" style="background:#e0524f;">{crit_initial}</div>
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
            verdict_color = "#e0524f" if label == "FAKE" else "#1fa971"
            st.markdown(
                f"""
                <div class="final-verdict-card" style="border-color:{verdict_color}66;">
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
    top_bar("Drejtësia & Bias-i i Modelit")
    st.markdown(
        '<p class="app-subtitle">Rezultatet e mëposhtme vijnë drejtpërdrejt nga vlerësimi i fairness-it '
        'i kryer në punimin e diplomës (Kreu III–IV), mbi 1.192 artikuj shqip. Janë gjetjet reale që '
        'qëndrojnë pas kontekstit të fairness-it që sheh te Analiza e Detajuar.</p>',
        unsafe_allow_html=True,
    )
 
    st.markdown(
        f"""<div class="stat-pill-row">
            <div class="stat-pill"><div class="stat-pill-value">95,5%</div><div class="stat-pill-label">Accuracy më e lartë (XLM-R, few-shot)</div></div>
            <div class="stat-pill"><div class="stat-pill-value">1.192</div><div class="stat-pill-label">Artikuj në grupin e testuar</div></div>
            <div class="stat-pill"><div class="stat-pill-value">Shëndetësi</div><div class="stat-pill-label">Grupi me normën më të lartë gabimi</div></div>
        </div>""",
        unsafe_allow_html=True,
    )
 
    st.markdown('<div class="section-label">Performanca e modeleve</div>', unsafe_allow_html=True)
    chart_card(
        "static/accuracy_models.png",
        "Katër modele (bazë TF-IDF+Logistic Regression, mBERT, XLM-R, mT5) u trajnuan në anglisht dhe u "
        "rregulluan më tej (<i>few-shot</i>) me 2.772 shembuj shqip. XLM-R doli modeli më i saktë mbi 594 "
        "artikuj testimi shqip.",
    )
 
    st.markdown('<div class="section-label">Drejtësia ndër-gjuhësore (zero-shot, EN→AL)</div>', unsafe_allow_html=True)
    chart_card(
        "static/fairness_gjuhesor.png",
        'Pa asnjë të dhënë shqipe (vetëm transferim nga anglishtja), XLM-R duket "më i drejtë" sipas Equal '
        'Opportunity Gap (0,003) — por kjo është artificiale: modeli kishte kolapsuar duke parashikuar "fake" '
        "për çdo artikull. mBERT (EOG 0,212) dhe mT5 (EOG 0,488) pasqyrojnë hendekë realë mes gjuhëve. Kjo është "
        "arsyeja pse EOG nuk duhet lexuar kurrë i vetëm, pa metrika shoqëruese.",
    )
 
    st.markdown('<div class="section-label">Bias sipas gjatësisë së artikullit</div>', unsafe_allow_html=True)
    chart_card(
        "static/bias_gjatesia.png",
        "XLM-R few-shot, artikujt shqip të ndarë në tri grupe sipas numrit të fjalëve. Norma e gabimit (FNR) "
        "rritet ndjeshëm te artikujt e gjatë — lajmet e rreme të gjata mbeten më shpesh të paidentifikuara.",
    )
 
    st.markdown('<div class="section-label">Bias sipas temës — i njëjti kontekst që sheh te Analiza</div>', unsafe_allow_html=True)
    chart_card(
        "static/bias_tema.png",
        "Kategoria <b>Shëndetësi</b> ka recall dukshëm më të ulët se pjesa tjetër — modeli mbështetet shumë te "
        "fjalori mjekësor i specializuar, më pak i pranishëm gjatë pre-trajnimit. Kjo është pikërisht statistika "
        "që përdor funksioni <code>fairness_context()</code> për t'i dhënë kontekst çdo analize në kohë reale.",
    )
 
    st.info("Për metodologjinë e plotë (si u llogaritën FPR/FNR/EOG dhe kufizimet e tyre), shih Kreun III–IV të punimit të diplomës.")
