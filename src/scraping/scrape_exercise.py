# src/scraping/exrx_bulk_scraper.py
# -*- coding: utf-8 -*-
"""
ExRx end-to-end scraper
=======================
Pipeline completa:
1) Crawling Exercise Directory → macro-categorie
2) Parsing di ogni pagina macro → righe (macro, micro, implement, name, url)
3) Build catalog per exercise_url
4) Dedup per slug (es. BWSquat unisce GluteusMaximus/Quadriceps)
5) Scraping dettagliato per ogni esercizio:
   - sections (classification/prep/execution/comments)
   - main_muscles_involved / stabilizers
   - tools (implement dedotto dallo slug)
   - download GIF (thumbnail) e MP4 (HLS via ffmpeg)
6) Salvataggi:
   data/benchmark/gif/   → GIF
   data/benchmark/clip/  → MP4
   data/benchmark/       → <slug>.json, _all_exercises.json, _dedup_index.json

Requisiti:
- playwright (con Chromium installato), beautifulsoup4, aiohttp
- (opzionale) opencv-python per validare primo frame video
- ffmpeg nel PATH
"""

from __future__ import annotations

import os
import re
import json
import shutil
import asyncio
import logging
from tqdm import tqdm
import subprocess
from dataclasses import dataclass, field
from typing import Any, Dict, Tuple, Optional, List
from pathlib import Path
from urllib.parse import urljoin, urlparse, urldefrag

import aiohttp
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright

# ------------------------------------------------------------
# Percorsi (root = cartella che contiene 'src' e 'data')
# ------------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]             # .../<root>/src/scraping/exrx_bulk_scraper.py
DATA_DIR = PROJECT_ROOT / "data" / "benchmark"
GIF_DIR = DATA_DIR / "gif"
CLIP_DIR = DATA_DIR / "clip"
JSON_DIR = DATA_DIR / "scraped_json"

GIF_DIR.mkdir(parents=True, exist_ok=True)
CLIP_DIR.mkdir(parents=True, exist_ok=True)
JSON_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# Logging
# ------------------------------------------------------------
logger = logging.getLogger("exrx_scraper")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(h)

# ------------------------------------------------------------
# Utils comuni
# ------------------------------------------------------------
SPACE = re.compile(r"\s+")
def norm(x: str) -> str:
    """Normalizza spazi e trim."""
    return SPACE.sub(" ", (x or "").strip())

def file_ok(path: str, min_bytes: int = 1024) -> bool:
    """True se il file esiste ed è almeno min_bytes."""
    try:
        return os.path.exists(path) and os.path.getsize(path) >= min_bytes
    except Exception:
        return False

def can_read_first_frame(video_path: str) -> bool:
    """
    Tenta di leggere il primo frame (richiede opencv). Se non disponibile,
    ritorna True se il file esiste e supera una soglia minima.
    """
    try:
        import cv2  # lazy import
        cap = cv2.VideoCapture(video_path)
        ok, frame = cap.read()
        cap.release()
        return bool(ok and frame is not None)
    except Exception:
        return file_ok(video_path, 50_000)

def slug_from_url(url: str) -> str:
    """Ritorna l’ultimo segmento del path (slug), utile per nomi file."""
    return url.rstrip("/").rsplit("/", 1)[-1]

def _ua() -> str:
    """User-Agent per Playwright e richieste HLS."""
    return ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36")

def _ffmpeg_bin() -> Optional[str]:
    """Path a ffmpeg se nel PATH, altrimenti None."""
    return shutil.which("ffmpeg")

# ------------------------------------------------------------
# 1) Fetch HTML generico (Playwright) e con token video sulla pagina esercizio
# ------------------------------------------------------------
async def fetch_html(url: str) -> str:
    """
    Scarica HTML di una pagina (macro o directory) con Playwright.
    - Attendi il DOM per contenuti popolati server-side, evita JS pesante.
    """
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        ctx = await browser.new_context(user_agent=_ua())
        page = await ctx.new_page()
        await page.goto(url, wait_until="domcontentloaded")
        await page.wait_for_timeout(1000)
        html = await page.content()
        await ctx.close()
        await browser.close()
        return html

async def fetch_page_with_token(url: str) -> Tuple[str, Optional[str]]:
    """
    Apre la pagina esercizio e restituisce:
    - html: contenuto completo
    - token: token per exrx.glorb.com/api/video/{token}/{fID} (se disponibile)
    """
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        ctx = await browser.new_context(user_agent=_ua())
        page = await ctx.new_page()
        await page.goto(url, wait_until="domcontentloaded")
        await page.wait_for_timeout(1000)

        html = await page.content()

        token = None
        try:
            token = await page.evaluate("""(async () => {
                try {
                    const r = await fetch('/api/video/token', {cache:'no-store'});
                    const j = await r.json();
                    return (j && j.token) ? j.token : null;
                } catch(e) { return null; }
            })()""")
        except Exception:
            token = None

        await ctx.close()
        await browser.close()
        return html, token

# ------------------------------------------------------------
# 2) Parsing Directory → macro categorie
# ------------------------------------------------------------
BASE = "https://exrx.net"

def last_path_key(u: str) -> str:
    p = urlparse(u)
    return p.path.rstrip("/").split("/")[-1]  # es. ShouldWt

PATH2MACRO = {
    "NeckWt": "Neck",
    "ShouldWt": "Shoulders",
    "ArmWt": "Upper Arms",
    "ForeArmWt": "Forearms",
    "BackWt": "Back",
    "ChestWt": "Chest",
    "WaistWt": "Waist",
    "HipsWt": "Hips",
    "ThighWt": "Thighs",
    "CalfWt": "Calves",
}
MACRO_NAMES = set(PATH2MACRO.values())

async def parse_macro_categories(dir_url=f"{BASE}/Lists/Directory") -> List[Dict[str, str]]:
    """
    Raccoglie le macro-categorie (nome e URL della lista esercizi) partendo dalla Directory.
    Deduplica i link e preferisce nomi canonici definiti in PATH2MACRO.
    """
    html = await fetch_html(dir_url)
    soup = BeautifulSoup(html, "lxml")

    candidates = []
    for a in soup.select("a[href]"):
        href = a.get("href")
        if not href:
            continue
        absu = urljoin(dir_url, href)
        if "/Lists/ExList/" not in absu:
            continue
        key = last_path_key(absu)              # es. ShouldWt
        url_clean = urldefrag(absu)[0]         # senza #fragment
        txt = norm(a.get_text())
        candidates.append((key, url_clean, txt))

    macros = {}
    for key, url_clean, txt in candidates:
        canonical = PATH2MACRO.get(key)
        display = canonical or txt
        if key not in macros:
            macros[key] = {"macro_key": key, "macro_name": display, "list_url": url_clean}
        else:
            if macros[key]["macro_name"] not in MACRO_NAMES and display in MACRO_NAMES:
                macros[key]["macro_name"] = display
            macros[key]["list_url"] = url_clean

    out = sorted(macros.values(), key=lambda d: d["macro_name"].lower())
    return out

# ------------------------------------------------------------
# 3) Parsing pagina macro → righe esercizi (con varianti)
# ------------------------------------------------------------
IMPLE_SET = {
    "barbell","body weight","bodyweight","cable","dumbbell","kettlebell",
    "lever","machine","smith","smith machine","suspended","self-assisted",
    "stretch","dynamic stretch","plyometrics","band","medicine ball","sandbag",
    "sled","trap bar","trx"
}

def is_implement_heading(text: str) -> bool:
    """Riconosce un header di implement/attrezzo nei vari formati usati su ExRx."""
    t = norm(text).lower()
    if t.startswith("lever"):
        return True
    return any(k == t or k in t for k in IMPLE_SET)

def norm_impl(label: str) -> str:
    """Normalizza label di implement (es. 'Lever (selectorized)' → 'lever')."""
    l = norm(label).lower()
    if l.startswith("lever"):         return "lever"
    if "smith" in l:                  return "smith machine"
    if "body weight" in l or l=="bodyweight": return "bodyweight"
    for k in IMPLE_SET:
        if k in l:
            return k
    return l or "unknown"

def _anchor(li, base_url: str):
    """Estrae (testo, url assoluto) da un <li> contenente un <a>."""
    a = li.find("a", href=True)
    if not a:
        return None, None
    return norm(a.get_text()), urldefrag(urljoin(base_url, a["href"]))[0]

def _direct_text(el) -> str:
    """Testo diretto del nodo (senza figli). Utile per <li> contenitore."""
    if el is None: return ""
    txts = [t for t in el.find_all(string=True, recursive=False)]
    return norm(" ".join(txts))

async def parse_macro_page(list_url: str, macro_name: str) -> List[Dict[str, str]]:
    """
    Estrae righe esercizio da una pagina macro:
    - macro_group, micro_group, implement, exercise_name, exercise_url, list_url
    Gestisce varianti annidate (es. “Front Raise — Alternating / One Arm”).
    """
    html = await fetch_html(list_url)
    soup = BeautifulSoup(html, "lxml")

    records: List[Dict[str, str]] = []
    current_micro = None
    current_impl  = None
    processed_uls = set()

    def emit(name, url, micro, impl):
        if not name or not url:
            return
        records.append({
            "macro_group": macro_name,
            "micro_group": micro or "",
            "implement": impl or "",
            "exercise_name": name,
            "exercise_url": url,
            "list_url": list_url,
        })

    def walk_variants(parent_li, base_name, base_url, micro, impl):
        """Cammina nelle <ul> annidate sotto una variante e emette 'base — variante'."""
        ul = parent_li.find("ul")
        if not ul or id(ul) in processed_uls:
            return
        processed_uls.add(id(ul))
        for child in ul.find_all("li", recursive=False):
            vname, vurl = _anchor(child, list_url)
            if vname and vurl:
                composed = f"{base_name} — {vname}"
                emit(composed, vurl, micro, impl)
            if child.find("ul"):
                walk_variants(child, base_name if not vname else composed, vurl or base_url, micro, impl)

    for node in soup.find_all(["h1","h2","h3","h4","strong","ul","li"]):
        tag  = node.name.lower()
        text = norm(node.get_text())

        # euristica per micro-muscolo
        if tag in ("h1","h2","h3") and text:
            if any(w in text.lower() for w in (
                "deltoid","quadriceps","hamstrings","gluteus","pectoralis","abdominis",
                "obliques","trapezius","rhomboids","erector","calves","soleus",
                "gastrocnemius","forearm","biceps","triceps","back","hip","waist","neck"
            )):
                current_micro = text
                current_impl  = None
                continue

        # implement come header
        if tag in ("h3","h4","strong") and text and is_implement_heading(text):
            current_impl = norm_impl(text)
            continue

        # implement come <li> contenitore
        if tag == "li" and node.find("ul"):
            impl_label = _direct_text(node)
            if is_implement_heading(impl_label):
                current_impl = norm_impl(impl_label)
                ul = node.find("ul")
                if id(ul) not in processed_uls:
                    processed_uls.add(id(ul))
                    for li in ul.find_all("li", recursive=False):
                        name, url = _anchor(li, list_url)
                        if name and url:
                            emit(name, url, current_micro, current_impl)
                            if li.find("ul"):
                                walk_variants(li, name, url, current_micro, current_impl)
                continue

        # UL “libero” sotto un header implement
        if tag == "ul" and current_micro:
            if node.find_parent("li") is not None:
                continue
            prev_hdr = node.find_previous(lambda t: t.name in ("h3","h4","strong","h2","h1"))
            if not (prev_hdr and is_implement_heading(prev_hdr.get_text())):
                continue
            current_impl = norm_impl(prev_hdr.get_text())
            if id(node) in processed_uls:
                continue
            processed_uls.add(id(node))
            for li in node.find_all("li", recursive=False):
                name, url = _anchor(li, list_url)
                if name and url:
                    emit(name, url, current_micro, current_impl)
                    if li.find("ul"):
                        walk_variants(li, name, url, current_micro, current_impl)

    # dedup per (url, micro, implement, name)
    seen, out = set(), []
    for r in records:
        key = (r["exercise_url"], r["micro_group"], r["implement"], r["exercise_name"])
        if key in seen:
            continue
        seen.add(key); out.append(r)
    return out

# ------------------------------------------------------------
# 4) Build catalog da righe e dedup per slug
# ------------------------------------------------------------
from collections import defaultdict

def build_catalog(rows: List[Dict[str, str]]) -> Dict[str, Dict[str, Any]]:
    """
    Collassa le righe in un indice per exercise_url:
      - name (sceglie il più descrittivo)
      - implements (set)
      - macro_groups (set)
      - micro_groups (set)
      - variants_by_impl (implement -> set di nomi completi)
      - list_urls (set)
    """
    cat: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        url = r["exercise_url"]
        name = r["exercise_name"]
        impl = r["implement"] or ""
        micro = r["micro_group"] or ""
        macro = r["macro_group"] or ""
        lst  = r["list_url"]

        doc = cat.get(url, {
            "name": name,
            "exercise_url": url,
            "implements": set(),
            "macro_groups": set(),
            "micro_groups": set(),
            "variants_by_impl": defaultdict(set),
            "list_urls": set(),
        })

        # tieni il nome più ricco
        if len(name) > len(doc["name"]):
            doc["name"] = name

        if impl:
            doc["implements"].add(impl)
            doc["variants_by_impl"][impl].add(name)
        else:
            doc["variants_by_impl"]["unknown"].add(name)

        if macro: doc["macro_groups"].add(macro)
        if micro: doc["micro_groups"].add(micro)
        if lst:   doc["list_urls"].add(lst)

        cat[url] = doc

    # set → liste ordinate
    for d in cat.values():
        d["implements"]    = sorted(d["implements"])
        d["macro_groups"]  = sorted(d["macro_groups"])
        d["micro_groups"]  = sorted(d["micro_groups"])
        d["list_urls"]     = sorted(d["list_urls"])
        d["variants_by_impl"] = {k: sorted(v) for k, v in d["variants_by_impl"].items()}
    return cat

def exrx_canonical_key(ex_url: str):
    """
    Chiave canonica ('WeightExercises', slug) per unire pagine duplicate
    (es. BWSquat sotto GluteusMaximus e Quadriceps).
    """
    p = urlparse(ex_url)
    parts = [x for x in p.path.split("/") if x]
    if not parts:
        return None
    slug = parts[-1]
    first = parts[0]
    return (first, slug)

def merge_docs(a: dict, b: dict) -> dict:
    """Unisci due documenti dello stesso esercizio."""
    out = dict(a)
    if len(b.get("name","")) > len(out.get("name","")):
        out["name"] = b["name"]
    for k in ("implements","macro_groups","micro_groups","list_urls"):
        sa = set(out.get(k, []))
        sb = set(b.get(k, []))
        out[k] = sorted(sa | sb)
    vbi_a = out.get("variants_by_impl", {})
    vbi_b = b.get("variants_by_impl", {})
    merged = {}
    for impl in set(vbi_a.keys()) | set(vbi_b.keys()):
        merged[impl] = sorted(set(vbi_a.get(impl, [])) | set(vbi_b.get(impl, [])))
    out["variants_by_impl"] = merged
    return out

def dedupe_by_slug(catalog: Dict[str, Dict[str, Any]]) -> Dict[Tuple[str,str], Dict[str, Any]]:
    """
    Collassa il catalog per ('WeightExercises', slug).
    Restituisce un dict indicizzato dalla chiave canonica, con:
      - exercise_url (preferito)
      - _all_urls (lista URL uniti)
      - macro_groups, micro_groups, implements, name, ...
    """
    buckets: Dict[Tuple[str,str], Dict[str, Any]] = {}

    for url, doc in catalog.items():
        key = exrx_canonical_key(url)
        if key is None:
            key = ("other", url.rsplit("/",1)[-1])
        if key not in buckets:
            buckets[key] = {
                **doc,
                "exercise_url": url,
                "_all_urls": [url],
            }
        else:
            buckets[key] = merge_docs(buckets[key], doc)
            buckets[key]["_all_urls"] = sorted(set(buckets[key].get("_all_urls", []) + [url]))

    return buckets

# ------------------------------------------------------------
# 5) Parser esercizio (sections/muscles/tools) e media (GIF/MP4)
# ------------------------------------------------------------
def text_after_h2(soup: BeautifulSoup, label: str) -> List[BeautifulSoup]:
    """Raccoglie nodi (p/li/ul/ol/table/div/strong) dopo un <h2> con titolo `label`."""
    h2 = soup.find(lambda t: t.name == "h2" and label.lower() in t.get_text(strip=True).lower())
    if not h2:
        return []
    nodes = []
    for sib in h2.find_all_next():
        if sib == h2:
            continue
        if sib.name == "h2":
            break
        if sib.name in ("p","ul","ol","li","table","div","strong"):
            nodes.append(sib)
    return nodes

def parse_classification_from_table(nodes: List[BeautifulSoup]) -> Dict[str, Optional[str]]:
    """Estrae Utility/Mechanics/Force da una tabella sotto 'Classification'."""
    out = {"utility": None, "mechanics": None, "force": None}
    for n in nodes:
        if n.name == "table":
            for tr in n.find_all("tr"):
                cells = [norm(td.get_text(" ")) for td in tr.find_all(["td","th"])]
                if len(cells) >= 2:
                    k = cells[0].lower().rstrip(":")
                    v = cells[-1]
                    if k == "utility":  out["utility"]  = v
                    if k == "mechanics":out["mechanics"]= v
                    if k == "force":    out["force"]    = v
    return out

def parse_instructions(nodes: List[BeautifulSoup]) -> Tuple[str, str, str]:
    """
    Estrae Preparation/Execution (e eventuali Comments inline) dalla sezione 'Instructions'.
    Ritorna (preparation, execution, comments_inline).
    """
    prep, execu, comm = [], [], []
    bucket = None
    for n in nodes:
        if n.name == "p":
            strong = n.find("strong")
            if strong:
                label = norm(strong.get_text()).lower()
                if label.startswith("preparation"):
                    bucket = "prep"; continue
                if label.startswith("execution"):
                    bucket = "exec"; continue
                if label.startswith("comments"):
                    bucket = "comm"; continue
            txt = norm(n.get_text(" "))
            if not txt: continue
            if bucket == "prep":   prep.append(txt)
            elif bucket == "exec": execu.append(txt)
            elif bucket == "comm": comm.append(txt)
            else:
                if not prep and not execu and not comm:
                    prep.append(txt)
                else:
                    execu.append(txt)
        elif n.name in ("ul","ol"):
            items = [norm(li.get_text(" ")) for li in n.find_all("li")]
            if not items: continue
            joined = "\n".join(f"- {it}" for it in items)
            if bucket == "prep":   prep.append(joined)
            elif bucket == "exec": execu.append(joined)
            elif bucket == "comm": comm.append(joined)
    return "\n".join(prep).strip(), "\n".join(execu).strip(), "\n".join(comm).strip()

def parse_comments(soup: BeautifulSoup) -> str:
    """Estrae la sezione 'Comments' (testo e liste)."""
    nodes = text_after_h2(soup, "comments")
    parts = []
    for n in nodes:
        if n.name == "p":
            t = norm(n.get_text(" "))
            if t: parts.append(t)
        elif n.name in ("ul","ol"):
            items = [norm(li.get_text(" ")) for li in n.find_all("li")]
            if items: parts.append("\n".join(f"- {it}" for it in items))
    return "\n".join(parts).strip()

def parse_muscles(soup: BeautifulSoup) -> Dict[str, List[str]]:
    """
    Estrae la sezione Muscles in tutte le sue varianti (Target/Synergists/Stabilizers/Dynamic Stabilizers).
    """
    nodes = []
    for key in ("muscles", "muscle", "primary muscles", "main muscles involved"):
        nodes = text_after_h2(soup, key)
        if nodes:
            break
    muscles = {"target": [], "synergists": [], "stabilizers": [], "dynamic_stabilizers": []}
    label = None
    for n in nodes:
        if n.name == "p":
            st = n.find("strong")
            if st:
                lab = norm(st.get_text()).lower()
                if "target" in lab:               label = "target"; continue
                if "synergist" in lab:            label = "synergists"; continue
                if lab.startswith("dynamic"):     label = "dynamic_stabilizers"; continue
                if "stabilizer" in lab:           label = "stabilizers"; continue
        if n.name == "ul" and label:
            items = [norm(li.get_text(" ")) for li in n.find_all("li")]
            muscles[label] = items
    return muscles

def muscles_flat_lists(m: Dict[str, List[str]]) -> Tuple[List[str], List[str]]:
    """
    Appiattisce i muscoli: Target+Synergists → main_muscles_involved,
    Stabilizers+Dynamic Stabilizers → stabilizers.
    """
    target = m.get("target", []) or []
    synerg = m.get("synergists", []) or []
    stabs  = (m.get("stabilizers", []) or []) + (m.get("dynamic_stabilizers", []) or [])
    def dedup(seq):
        seen, out = set(), []
        for x in seq:
            if x not in seen:
                seen.add(x); out.append(x)
        return out
    return dedup(target + synerg), dedup(stabs)

def tools_from_slug(slug: str) -> List[str]:
    """
    Deduci l'implement principale dallo slug ExRx (BW/BB/DB/CB/KB/LV/SM/SL).
    Ritorna lista di stringhe (es. ["bodyweight"]).
    """
    s = slug.upper()
    implement = None
    if s.startswith("BW"): implement = "bodyweight"
    elif s.startswith("BB"): implement = "barbell"
    elif s.startswith("DB"): implement = "dumbbell"
    elif s.startswith("CB"): implement = "cable"
    elif s.startswith("KB"): implement = "kettlebell"
    elif s.startswith("LV"): implement = "lever"
    elif s.startswith("SM"): implement = "smith machine"
    elif s.startswith("SL"): implement = "sled"
    return [implement] if implement else []

def parse_exrx_exercise_v3(html: str, url: str) -> Dict[str, Any]:
    """
    Parser contenuti della pagina esercizio:
    - name, classification, preparation/execution/comments,
    - main_muscles_involved + stabilizers (liste piatte),
    - tools (lista di stringhe).
    """
    soup = BeautifulSoup(html, "lxml")

    h1 = soup.find("h1", class_="page-title") or soup.find("h1")
    name = norm(h1.get_text()) if h1 else "Exercise"

    class_nodes = text_after_h2(soup, "classification")
    classification = parse_classification_from_table(class_nodes)

    instr_nodes = text_after_h2(soup, "instructions")
    preparation, execution, inline_comments = parse_instructions(instr_nodes)

    comments = parse_comments(soup) or inline_comments

    muscles_raw = parse_muscles(soup)
    main_list, stabs_list = muscles_flat_lists(muscles_raw)

    slug = slug_from_url(url)
    tools = tools_from_slug(slug)

    return {
        "name": name,
        "url": url,
        "tools": tools,  # lista di stringhe (es. ["bodyweight"])
        "sections": {
            "classification": classification,
            "preparation": preparation,
            "execution": execution,
            "comments": comments,
        },
        "main_muscles_involved": main_list,
        "stabilizers": stabs_list,
    }

# ---- Media: GIF + Video ------------------------------------------------------
def first_non_logo_thumb(html: str, base_url: str) -> Optional[str]:
    """
    Sceglie una thumbnail “buona”:
    - <meta property="og:image">, poi <meta name="thumbnail">
    - fallback: prima <img> non-logo in pagina
    """
    soup = BeautifulSoup(html, "lxml")
    og = soup.find("meta", attrs={"property":"og:image"})
    if og and og.get("content"):
        return og["content"]
    tn = soup.find("meta", attrs={"name":"thumbnail"})
    if tn and tn.get("content"):
        return tn["content"]
    for img in soup.find_all("img"):
        src = img.get("src") or ""
        if "logo" in src.lower():
            continue
        return src if src.startswith("http") else urljoin(base_url, src)
    return None

async def download_binary(url: str, dest: Path) -> bool:
    """
    Scarica un binario generico su disco.
    Ritorna True/False.
    """
    try:
        async with aiohttp.ClientSession() as sess:
            async with sess.get(url, timeout=120) as r:
                r.raise_for_status()
                data = await r.read()
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        return True
    except Exception as e:
        logger.warning(f"download_binary error: {e}")
        return False

async def save_thumb(html: str, url: str) -> Dict[str, str]:
    """
    Scarica la GIF/thumbnail in data/benchmark/gif/.
    Ritorna: {"thumb_path": "data/benchmark/gif/<name>.gif"} (relativo alla root del progetto).
    """
    out: Dict[str, str] = {}
    thumb_url = first_non_logo_thumb(html, url)
    if not thumb_url:
        return out
    name = (thumb_url.rsplit("/",1)[-1].split("?")[0]) or "thumb.bin"
    dest = GIF_DIR / name
    ok = await download_binary(thumb_url, dest)
    if ok and file_ok(str(dest), 200):
        out["thumb_path"] = str(dest.relative_to(PROJECT_ROOT))
    return out

def parse_fid_from_script(html: str) -> Optional[int]:
    """Estrae const fID = <num>; dal blocco JS del player."""
    m = re.search(r"const\s+fID\s*=\s*(\d+)\s*;", html)
    return int(m.group(1)) if m else None

async def save_exrx_clip(html: str, token: Optional[str], page_url: str) -> Dict[str, Any]:
    """
    Scarica l’MP4 del player ExRx:
    - se token & fID: prova https://exrx.glorb.com/api/video/{token}/{fID} (+ master.m3u8 / index.m3u8)
    - altrimenti fallback al <video><source> (se presente)
    - usa ffmpeg per HLS → MP4 (copia → altrimenti transcode)
    Salva in data/benchmark/clip/<fid|slug>.mp4
    """
    out: Dict[str, Any] = {}
    fid = parse_fid_from_script(html)

    if token and fid:
        base_clip = f"https://exrx.glorb.com/api/video/{token}/{fid}"
        out_name = f"{fid}.mp4"
    else:
        soup = BeautifulSoup(html, "lxml")
        vid = soup.find("video")
        src = None
        if vid:
            src = vid.get("src")
            if not src:
                se = vid.find("source")
                if se and se.get("src"):
                    src = se["src"]
        if not src:
            return out
        base_clip = src if src.startswith("http") else urljoin(page_url, src)
        out_name = f"{slug_from_url(page_url)}.mp4"

    candidates = [
        base_clip,
        base_clip.rstrip("/") + "/master.m3u8",
        base_clip.rstrip("/") + "/index.m3u8",
    ]

    ffmpeg = _ffmpeg_bin()
    if not ffmpeg:
        logger.warning("ffmpeg non trovato nel PATH: impossibile salvare HLS.")
        return out

    parsed = urlparse(page_url)
    referer = f"{parsed.scheme}://{parsed.netloc}"
    headers = [f"Referer: {referer}", f"Origin: {referer}", "Accept: */*"]
    headers_str = "\r\n".join(headers) + "\r\n"
    ua = _ua()

    out_path = CLIP_DIR / out_name

    def try_ffmpeg(input_url: str) -> bool:
        """Tenta prima copia stream, poi transcodifica; pulisce file troppo piccoli."""
        # 1) copia
        cmd_copy = [
            ffmpeg, "-y", "-loglevel", "warning",
            "-user_agent", ua, "-headers", headers_str,
            "-i", input_url, "-c", "copy", "-bsf:a", "aac_adtstoasc",
            str(out_path)
        ]
        p = subprocess.run(cmd_copy, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if p.returncode == 0 and file_ok(str(out_path), 50_000):
            return True

        # 2) transcode
        cmd_trans = [
            ffmpeg, "-y", "-loglevel", "warning",
            "-user_agent", ua, "-headers", headers_str,
            "-i", input_url,
            "-vf", "scale=640:-2", "-r", "25",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast", "-crf", "22",
            "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart",
            str(out_path)
        ]
        p2 = subprocess.run(cmd_trans, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if p2.returncode == 0 and file_ok(str(out_path), 50_000):
            return True

        try:
            if out_path.exists() and out_path.stat().st_size < 50_000:
                out_path.unlink(missing_ok=True)
        except Exception:
            pass
        return False

    ok = any(try_ffmpeg(c) for c in candidates)
    if not ok:
        logger.info("❌ Download video fallito o file troppo piccolo.")
        return out

    out["clip_path"] = str(out_path.relative_to(PROJECT_ROOT))
    out["clip_can_read_first_frame"] = can_read_first_frame(str(out_path))
    return out

# ------------------------------------------------------------
# 6) Orchestrazione singolo esercizio + bulk
# ------------------------------------------------------------
def validate_doc(doc: Dict[str, Any]) -> Dict[str, bool]:
    """
    Valida campi essenziali che ci interessano monitorare.
    """
    sections = doc.get("sections", {})
    prep_ok = bool(sections.get("preparation"))
    exec_ok = bool(sections.get("execution"))
    muscles_ok = bool(doc.get("main_muscles_involved"))
    video_ok = bool(doc.get("media", {}).get("clip_path"))
    thumb_ok = bool(doc.get("media", {}).get("thumb_path"))
    return {
        "prep_ok": prep_ok,
        "exec_ok": exec_ok,
        "muscles_ok": muscles_ok,
        "video_ok": video_ok,
        "thumb_ok": thumb_ok,
    }

@dataclass
class SafeCounters:
    """
    Contatori globali thread-safe (async) per riassunti e progress,
    da usare anche con scraping concorrente.
    """
    total: int = 0
    ok_full: int = 0
    missing_prep: int = 0
    missing_exec: int = 0
    missing_muscles: int = 0
    missing_video: int = 0
    missing_thumb: int = 0
    failed: int = 0
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def bump_from_flags(self, flags: Dict[str, bool]):
        async with self.lock:
            self.total += 1
            if all(flags.values()):
                self.ok_full += 1
            if not flags["prep_ok"]:
                self.missing_prep += 1
            if not flags["exec_ok"]:
                self.missing_exec += 1
            if not flags["muscles_ok"]:
                self.missing_muscles += 1
            if not flags["video_ok"]:
                self.missing_video += 1
            if not flags["thumb_ok"]:
                self.missing_thumb += 1

    async def bump_failed(self):
        async with self.lock:
            self.failed += 1

    async def snapshot(self) -> dict:
        """Ritorna una copia atomica dei contatori (per mostrarli in tqdm)."""
        async with self.lock:
            return {
                "total": self.total,
                "ok_full": self.ok_full,
                "miss_prep": self.missing_prep,
                "miss_exec": self.missing_exec,
                "miss_musc": self.missing_muscles,
                "miss_vid": self.missing_video,
                "miss_thu": self.missing_thumb,
                "failed": self.failed,
            }
async def process_one_exercise(
    canon_key: Any,
    meta: Dict[str, Any],
    sem: asyncio.Semaphore,
    counters: SafeCounters,
) -> Optional[Tuple[Dict[str, Any], Dict[str, bool]]]:
    """
    Scraping completo per un esercizio:
    - fetch HTML + token
    - parse contenuti
    - salva GIF e MP4
    - costruisce record finale con output semplificato
    - salva JSON per-esercizio
    - aggiorna contatori globali
    """
    async with sem:
        pref_url = meta.get("preferred_url") or (meta.get("all_urls") or [None])[0]
        if not pref_url:
            logger.warning(f"[SKIP] {canon_key}: nessun URL.")
            await counters.bump_failed()
            return None

        # 1) fetch pagina + token
        try:
            html, token = await fetch_page_with_token(pref_url)
        except Exception as e:
            logger.error(f"[ERR] fetch fallito: {pref_url} → {e}")
            await counters.bump_failed()
            return None

        # 2) parse pagina (contenuti testuali)
        try:
            parsed = parse_exrx_exercise_v3(html, pref_url)
        except Exception as e:
            logger.error(f"[ERR] parse fallito: {pref_url} → {e}")
            await counters.bump_failed()
            return None

        # 3) salva media (thumb + clip)
        media = {}
        try:
            media.update(await save_thumb(html, pref_url))
        except Exception as e:
            logger.warning(f"thumb fallita: {pref_url} → {e}")
        try:
            media.update(await save_exrx_clip(html, token, pref_url))
        except Exception as e:
            logger.warning(f"clip fallita: {pref_url} → {e}")

        # Normalizza struttura media (se manca qualcosa resta None)
        media = {
            "thumb_path": media.get("thumb_path"),
            "clip_path": media.get("clip_path"),
            "clip_can_read_first_frame": media.get("clip_can_read_first_frame"),
        }

        # 4) record finale con schema semplificato richiesto
        canonical_key = canon_key if isinstance(canon_key, (str, int)) else list(canon_key)
        record = {
            "canonical_key": canonical_key,
            "name": parsed.get("name") or meta.get("name"),
            "url": parsed.get("url") or pref_url,
            "tools": parsed.get("tools", []),  # ← array di stringhe
            "sections": parsed.get("sections", {}),
            "main_muscles_involved": parsed.get("main_muscles_involved", []),
            "stabilizers": parsed.get("stabilizers", []),
            "media": media
        }

        # fallback muscoli: se vuoto, usa i micro_groups dal dedup
        if not record.get("main_muscles_involved"):
            mg = meta.get("micro_groups", [])
            record["main_muscles_involved"] = mg[:] if mg else []

        # 5) validazione + log sintetico + contatori
        flags = validate_doc(record)
        flag_labels = {
            "prep_ok": "preparation",
            "exec_ok": "execution",
            "muscles_ok": "muscles",
            "video_ok": "video",
            "thumb_ok": "thumbnail"
        }
        missing = [flag_labels.get(k, k) for k, ok in flags.items() if not ok]
        if missing:
            logger.info(f"[INFO] {record['name'] or 'N/A'} → missing: {', '.join(missing)}")
        else:
            logger.info(f"[OK]   {record['name'] or 'N/A'}")

        await counters.bump_from_flags(flags)

        # 6) salva JSON per esercizio (schema semplificato)
        fname = slug_from_url(pref_url) or "exercise"
        (JSON_DIR / f"{fname}.json").write_text(json.dumps(record, ensure_ascii=False, indent=2))

        return record, flags

async def run_bulk_from_dedup(
    dedup: Dict[Any, Dict[str, Any]],
    max_concurrency: int = 3
):
    """
    Esegue lo scraping sull’intero dedup con barra di progresso tipo tqdm.
    Scrive JSON per-esercizio e master _all_exercises.json.
    Ritorna (master_records, summary_dict).
    """
    sem = asyncio.Semaphore(max_concurrency)
    counters = SafeCounters()

    # creiamo tutte le coroutine, ma le eseguiamo monitorandole con as_completed
    coros = [
        process_one_exercise(k, v, sem, counters)
        for k, v in dedup.items()
    ]
    tasks = [asyncio.create_task(c) for c in coros]

    master: List[Dict[str, Any]] = []
    total = len(tasks)

    # tqdm che avanza al completamento di ogni task
    with tqdm(total=total, desc="Scraping", unit="ex") as pbar:
        for fut in asyncio.as_completed(tasks):
            try:
                res = await fut
                if isinstance(res, tuple):
                    record, _flags = res
                    master.append(record)
                elif res is None:
                    # già conteggiato come failed dentro process_one_exercise
                    pass
            except Exception:
                # eventuale eccezione non gestita
                await counters.bump_failed()

            # aggiorna la barra
            pbar.update(1)
            # mostra un piccolo riassunto in tempo reale
            snap = await counters.snapshot()
            pbar.set_postfix(snap, refresh=False)

    # salva master
    (JSON_DIR / "_all_exercises.json").write_text(json.dumps(master, ensure_ascii=False, indent=2))

    # summary finale
    summary = {
        "total": counters.total,
        "ok_full": counters.ok_full,
        "missing_prep": counters.missing_prep,
        "missing_exec": counters.missing_exec,
        "missing_muscles": counters.missing_muscles,
        "missing_video": counters.missing_video,
        "missing_thumb": counters.missing_thumb,
        "failed": counters.failed,
    }

    logger.info("\n==== SUMMARY ====")
    for k, v in summary.items():
        logger.info(f"{k:>16}: {v}")
    logger.info(f"Master JSON → {JSON_DIR / '_all_exercises.json'}")

    return master, summary

# ------------------------------------------------------------
# 7) MAIN end-to-end: genera DEDUP e lancia scraping
# ------------------------------------------------------------
async def main(max_concurrency: int = 3, delay_between_macros: float = 0.6):
    """
    Esegue l'intera pipeline:
    - Scopre macro-categorie
    - Estrae esercizi da ogni macro
    - Costruisce catalog e dedup
    - Salva indice dedup
    - Scrapa in dettaglio ogni esercizio (concorrenza configurabile)
    """
    # 1) Macro-categorie
    macros = await parse_macro_categories()
    logger.info(f"Macro-categorie: {len(macros)}")

    # 2) Righe esercizi da TUTTE le macro
    all_rows: List[Dict[str, str]] = []
    for i, m in enumerate(macros, 1):
        try:
            rows = await parse_macro_page(m["list_url"], macro_name=m["macro_name"])
            all_rows.extend(rows)
            logger.info(f"[{i}/{len(macros)}] {m['macro_name']:<12} → {len(rows)} esercizi (tot {len(all_rows)})")
        except Exception as e:
            logger.error(f"[ERRORE] {m['macro_name']}: {e}")
        await asyncio.sleep(delay_between_macros)

    # 3) Catalog complessivo per URL
    catalog = build_catalog(all_rows)
    logger.info(f"Catalog (URL unici): {len(catalog)}")

    # 4) Dedup per slug ('WeightExercises', slug)
    buckets = dedupe_by_slug(catalog)
    logger.info(f"Dedup per slug: {len(buckets)}")

    # 5) Trasforma in struttura 'dedup' attesa da run_bulk_from_dedup
    dedup: Dict[Any, Dict[str, Any]] = {}
    for canon_key, doc in buckets.items():
        dedup[canon_key] = {
            "preferred_url": doc.get("exercise_url"),
            "all_urls": doc.get("_all_urls", []),
            "macro_groups": doc.get("macro_groups", []),
            "micro_groups": doc.get("micro_groups", []),
            "implements": doc.get("implements", []),
            "name": doc.get("name"),
        }

    # Salva indice dedup per ispezione
    (JSON_DIR / "_dedup_index.json").write_text(
        json.dumps({str(k): v for k, v in dedup.items()}, ensure_ascii=False, indent=2)
    )

    # 6) Scraping dettagliato + download media
    await run_bulk_from_dedup(dedup, max_concurrency=max_concurrency)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="ExRx end-to-end scraper")
    parser.add_argument("--concurrency", type=int, default=3, help="Numero massimo di esercizi in parallelo")
    args = parser.parse_args()
    asyncio.run(main(max_concurrency=args.concurrency))