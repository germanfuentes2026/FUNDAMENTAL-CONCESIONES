"""Extractor de estados contables SIN IA (lectura local del PDF) + relleno opcional con IA barata.

Flujo:
1. pypdf extrae el texto de cada página (gratis, local).
2. Se ubican las páginas del balance, estado de resultados y flujo de efectivo.
3. Regex sobre las líneas ("Total del activo  1.234.567  987.654") para sacar cada campo, ejercicio actual y comparativo.
4. Opcional: si quedan campos vacíos y el usuario lo activa, se manda a Claude SOLO el texto de esas 2-4 páginas
   (no el PDF) y se piden SOLO los campos faltantes, con un modelo chico y pocos tokens de salida.
"""
from __future__ import annotations

import io
import json
import os
import re
import unicodedata
from datetime import date

from pypdf import PdfReader

FIELDS = [
    "current_assets", "current_liabilities", "total_assets", "total_liabilities",
    "equity", "retained_earnings", "ebit", "sales",
    "net_income", "cfo", "long_term_debt", "cost_of_sales", "share_capital",
]
AI_MODEL = os.getenv("CLAUDE_MODEL", "claude-haiku-5-5")
AI_MAX_CHARS = 30000

MONTHS = {"enero": 1, "febrero": 2, "marzo": 3, "abril": 4, "mayo": 5, "junio": 6, "julio": 7,
          "agosto": 8, "septiembre": 9, "setiembre": 9, "octubre": 10, "noviembre": 11, "diciembre": 12}


# ───────────────────────── utilidades de texto/números ─────────────────────────
def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    return re.sub(r"[ \t\u00a0]+", " ", s)


_TOK = re.compile(r"\(?-?\s?\d[\d.,]*\)?|(?<=\s)-(?=\s|$)")
_NOTA = re.compile(r"\(?\bnotas?\b[^\d\n]{0,6}\d+(?:\s*(?:,|y|a|/)\s*\d+)*\)?", re.I)


def _val(raw: str):
    raw = raw.strip()
    neg = raw.startswith("(") or raw.startswith("-")
    s = re.sub(r"[()\-\s]", "", raw)
    if s == "":
        return 0.0
    if "," in s:
        s = s.replace(".", "").replace(",", ".")
    elif re.fullmatch(r"\d{1,3}(\.\d{3})+", s):
        s = s.replace(".", "")
    try:
        v = float(s)
    except ValueError:
        return None
    return -v if neg else v


def _split(line: str):
    line = _NOTA.sub(" ", line)
    toks = list(_TOK.finditer(line))
    if not toks:
        return _norm(line).strip(" .:$-"), None
    label = _norm(line[:toks[0].start()]).strip(" .:$-")
    raws = [t.group().strip() for t in toks]
    if len(raws) >= 3:
        raws = raws[-2:]
    elif len(raws) == 2 and re.fullmatch(r"\d{1,2}", raws[0]) and (
            "." in raws[1] or len(re.sub(r"\D", "", raws[1])) > 3):
        raws = raws[1:]  # el primero es el nº de nota
    vals = [_val(r) for r in raws]
    if len(vals) == 1:
        return label, (vals[0], None)
    return label, (vals[0], vals[1])


def _lines(text: str) -> list:
    raw = [l.strip() for l in text.splitlines() if l.strip()]
    out, i = [], 0
    while i < len(raw):
        l = raw[i]
        if (not re.search(r"\d", l) and i + 1 < len(raw)
                and not re.search(r"[A-Za-zÁ-ú]", raw[i + 1]) and re.search(r"\d", raw[i + 1])):
            l = l + " " + raw[i + 1]
            i += 1
        out.append(_split(l))
        i += 1
    return out


def _firstl(lines, pat, excl=None, start=0, end=None):
    rx = re.compile(pat)
    ex = re.compile(excl) if excl else None
    for lab, c in lines[start:end]:
        if c and rx.search(lab) and not (ex and ex.search(lab)):
            return lab, c
    return None, None


def _first(lines, pat, excl=None):
    return _firstl(lines, pat, excl)[1]


def _idx(lines, pat, need_nums=True):
    rx = re.compile(pat)
    for i, (lab, c) in enumerate(lines):
        if rx.search(lab) and (c or not need_nums):
            return i
    return None


def _sumcols(items):
    if not items:
        return None
    out = []
    for k in (0, 1):
        vs = [c[k] for c in items if c[k] is not None]
        out.append(sum(vs) if vs else None)
    return tuple(out)


def _op(a, b, fn):
    """Operación columna a columna; None si falta algún dato."""
    if a is None or b is None:
        return None
    return tuple(None if a[k] is None or b[k] is None else fn(a[k], b[k]) for k in (0, 1))


# ───────────────────────── extracción local ─────────────────────────
def _dates(norm_text: str):
    found = []
    for d, mon, y in re.findall(r"(\d{1,2})\s+de\s+(" + "|".join(MONTHS) + r")\s+de\s+(\d{4})", norm_text):
        found.append(date(int(y), MONTHS[mon], int(d)))
    for d, mo, y in re.findall(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b", norm_text):
        try:
            found.append(date(int(y), int(mo), int(d)))
        except ValueError:
            pass
    uniq = []
    for x in found:
        if x not in uniq:
            uniq.append(x)
    return uniq


def _prev_year(d: date) -> date:
    try:
        return d.replace(year=d.year - 1)
    except ValueError:
        return d.replace(year=d.year - 1, day=28)


def _company_name(texts) -> str | None:
    head = "\n".join(texts[:3])
    m = re.search(r"(?:denominaci[oó]n|raz[oó]n social)(?:\s+de\s+la\s+sociedad)?\s*:?\s*\n?\s*([^\n]{4,80})", head, re.I)
    if m:
        return m.group(1).strip(" .:")
    m = re.search(r"\b([A-ZÁÉÍÓÚÑ][A-ZÁÉÍÓÚÑ0-9 .,&'\-]{4,70}?\s(?:S\.?A\.?[A-Z.]*|S\.?R\.?L\.?))(?=\s|$)", head)
    return m.group(1).strip() if m else None


def _parse(texts):
    n = [_norm(t) for t in texts]
    bal_i = [i for i, t in enumerate(n) if re.search(r"total (del )?(activo|pasivo|patrimonio neto)\b", t)][:3]
    if not bal_i:
        con_texto = sum(1 for t in texts if len(t.strip()) > 50)
        raise ValueError(f"No encontré el estado de situación patrimonial. Páginas: {len(texts)}, con texto: {con_texto}. "
                         "Cargá los datos a mano en DATOS o activá la IA.")
    res_i = [i for i, t in enumerate(n)
             if re.search(r"ingresos|ventas", t)
             and re.search(r"resultado (neto )?del (ejercicio|periodo)|ganancia (neta )?del (ejercicio|periodo)|ganancia \(perdida\) del", t)
             and re.search(r"antes (del|de) impuesto|resultado operativo|costo", t)][:2]
    cf_i = [i for i, t in enumerate(n)
            if "flujo de efectivo" in t and re.search(r"actividades (operativas|de operacion)", t)][:2]

    L_bal = [x for i in bal_i for x in _lines(texts[i])]
    L_res = [x for i in res_i for x in _lines(texts[i])]
    L_cf = [x for i in cf_i for x in _lines(texts[i])]
    notes: list[str] = []

    # unidad
    head = " ".join(n[i] for i in bal_i[:1] + res_i[:1])
    mult = 1.0
    if re.search(r"en miles de pesos|expresad[oa]s? en miles|en miles de \$|cifras en miles", head):
        mult = 1e3
    elif re.search(r"en millones", head):
        mult = 1e6

    # fechas
    ds = _dates(head)
    if not ds:
        raise ValueError("No pude detectar la fecha de cierre del ejercicio en el balance.")
    cur_end = max(ds)
    prev = [d for d in ds if d.year == cur_end.year - 1]
    prior_end = max(prev) if prev else _prev_year(cur_end)

    V: dict[str, tuple | None] = {}
    V["total_assets"] = _first(L_bal, r"^total (del )?activo$")
    V["current_assets"] = _first(L_bal, r"^total (del )?activo corriente")
    V["current_liabilities"] = _first(L_bal, r"^total (del )?pasivo corriente")
    V["total_liabilities"] = _first(L_bal, r"^total (del )?pasivo$")
    V["equity"] = _first(L_bal, r"^total (del )?patrimonio neto|^patrimonio neto total")
    if V["total_liabilities"] is None:
        V["total_liabilities"] = _op(V["total_assets"], V["equity"], lambda a, e: a - e)
        if V["total_liabilities"]:
            notes.append("Pasivo total = activo − patrimonio neto.")

    V["sales"] = _first(L_res, r"^(ingresos?|ventas?)\b.*(servicios|peaje|operativ|ordinari|netas|concesi)|^ventas netas|^ingresos$|^ventas$",
                        r"costo|financier|otros")
    V["cost_of_sales"] = _first(L_res, r"^costos?\s+de\s+(los\s+|la\s+)?(servicios|ventas|explotacion|operacion|concesion)")
    V["net_income"] = (_first(L_res, r"^(resultado|ganancia|perdida)\s*(\(perdida\)\s*)?(neto|neta)?\s*(del\s+ejercicio|del\s+periodo)\b", r"antes|otro")
                       or _first(L_res, r"^(ganancia|perdida|resultado)\s*(\(perdida\)\s*)?(neta?)?\s*$"))

    ebit = _first(L_res, r"^(resultado|ganancia|utilidad|perdida)\b.*\b(operativ[oa]|de explotacion)\b", r"antes|actividades")
    if ebit is None:
        ebt = _first(L_res, r"^(resultado|ganancia|perdida)\b.*antes (del|de)\s+impuesto")
        flab, fin = _firstl(L_res, r"^resultados?\s+financieros?")
        _, rec = _firstl(L_res, r"recpam|poder adquisitivo")
        if ebt and fin:
            ebit = _op(ebt, fin, lambda a, b: a - b)
            if rec and "recpam" not in (flab or ""):
                ebit = _op(ebit, rec, lambda a, b: a - b)
            notes.append("EBIT estimado = resultado antes de impuesto − resultados financieros (verificar).")
    V["ebit"] = ebit

    V["cfo"] = _first(L_cf, r"(flujo|efectivo).*(generado|proveniente|utilizado|usado|neto).*actividades\s+(operativas|de\s+operacion)")

    # capital y resultados acumulados (sección patrimonio del balance)
    i_cap = _idx(L_bal, r"^capital\b(?!.*ajuste)")
    i_eq = _idx(L_bal, r"^total (del )?patrimonio neto")
    V["share_capital"] = L_bal[i_cap][1] if i_cap is not None else None
    V["retained_earnings"] = None
    if i_cap is not None and i_eq is not None and i_eq > i_cap:
        region = [(lab, c) for lab, c in L_bal[i_cap:i_eq] if c and not lab.startswith("total")]
        accum = [c for lab, c in region if re.search(r"resultados?\s+(no\s+asignados?|acumulados?)", lab)]
        resv = [c for lab, c in region if re.search(r"reserva", lab)]
        ejer = [c for lab, c in region if re.search(r"resultado\s+del\s+(ejercicio|periodo)", lab)]
        items = resv + accum + ([] if accum else ejer)
        V["retained_earnings"] = _sumcols(items)
        if items:
            notes.append("Resultados acumulados = reservas + resultados no asignados (verificar).")

    # deuda financiera no corriente
    i_nc = _idx(L_bal, r"^pasivo no corriente$", need_nums=False)
    i_tnc = _idx(L_bal, r"^total (del )?pasivo no corriente")
    V["long_term_debt"] = None
    if i_nc is not None and i_tnc is not None and i_tnc > i_nc:
        items = [c for lab, c in L_bal[i_nc + 1:i_tnc]
                 if c and not lab.startswith("total")
                 and re.search(r"prestamos|obligaciones negociables|deudas? financieras|financiaciones", lab)]
        V["long_term_debt"] = _sumcols(items) or (0.0, 0.0)

    recs = []
    for k, (end, col) in enumerate(((cur_end, "current"), (prior_end, "prior"))):
        rec = {f: (None if V.get(f) is None or V[f][k] is None else V[f][k] * mult) for f in FIELDS}
        recs.append({"period_end": end.isoformat(), "column": col, **rec})
    pages = sorted(set(bal_i + res_i + cf_i))
    return recs, pages, mult, notes, _company_name(texts)


# ───────────────────────── IA opcional (solo faltantes) ─────────────────────────
def _ai_fill(text: str, miss_cur: list, miss_prior: list, key: str, model: str) -> dict:
    import anthropic

    prompt = (
        "Del texto de estados contables argentinos (concesionaria vial) extraé SOLO estos campos. "
        f"Ejercicio actual: {miss_cur}. Ejercicio anterior (columna comparativa): {miss_prior}.\n"
        "Devolvé solo JSON: {\"current\": {campo: n}, \"prior\": {campo: n}}. Números tal como figuran "
        "(sin convertir unidades), pérdidas negativas, null si no está. No inventes.\n"
        "Definiciones: retained_earnings = reservas + resultados no asignados; ebit = resultado operativo antes de "
        "resultados financieros e impuesto; cfo = flujo de actividades operativas; long_term_debt = deudas financieras "
        "no corrientes; cost_of_sales en positivo; share_capital nominal sin ajuste.\n\nTEXTO:\n" + text[:AI_MAX_CHARS]
    )
    msg = anthropic.Anthropic(api_key=key).messages.create(
        model=model, max_tokens=800, messages=[{"role": "user", "content": prompt}])
    out = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
    a, b = out.find("{"), out.rfind("}")
    return json.loads(out[a:b + 1]) if a >= 0 and b >= 0 else {}


# ───────────────────────── API pública ─────────────────────────
def extract_pdf(pdf_bytes: bytes, source: str, company: str | None = None, use_ai: bool = False,
                api_key: str = "", ai_model: str = AI_MODEL, cache: dict | None = None):
    """Devuelve (registros, info). info = {"missing": [...], "ai": [...]}."""
    reader = PdfReader(io.BytesIO(pdf_bytes))
    texts = []
    for p in reader.pages:
        try:
            texts.append(p.extract_text() or "")
        except Exception:  # noqa: BLE001
            texts.append("")
    if sum(len(t.strip()) for t in texts) < 300:
        raise ValueError("El PDF no tiene texto seleccionable (parece escaneado). Cargá los datos a mano en DATOS.")

    periods, pages, mult, notes, detected = _parse(texts)
    ai_done: list[str] = []

    if use_ai and api_key:
        miss = [[f for f in FIELDS if p[f] is None] for p in periods]
        if any(miss):
            ck = (hash(pdf_bytes), tuple(map(tuple, miss)))
            cache = cache if cache is not None else {}
            if ck not in cache:
                text = "\n\n".join(texts[i] for i in pages)
                try:
                    cache[ck] = _ai_fill(text, miss[0], miss[1], api_key, ai_model)
                except Exception as exc:  # noqa: BLE001
                    notes.append(f"IA no disponible: {exc}")
                    cache[ck] = {}
            res = cache[ck]
            for k, name in enumerate(("current", "prior")):
                for f in miss[k]:
                    v = (res.get(name) or {}).get(f)
                    try:
                        v = None if v is None else float(v) * mult
                    except (TypeError, ValueError):
                        v = None
                    if v is not None:
                        periods[k][f] = v
                        if k == 0:
                            ai_done.append(f)
            if ai_done:
                notes.append("Completado con IA: " + ", ".join(ai_done) + ".")

    name = company or detected or "Sin nombre"
    recs = []
    for p in periods:
        if all(p[f] is None for f in FIELDS):
            continue
        recs.append({
            "company": name, "fy": int(p["period_end"][:4]), "period_end": p["period_end"],
            "origin": "own" if p["column"] == "current" else "comparative",
            "source": source, **{f: p[f] for f in FIELDS}, "notes": " ".join(notes),
        })
    if not recs:
        raise ValueError("No pude leer valores del balance. Cargalos a mano en DATOS o activá la IA.")
    missing = [f for f in FIELDS if periods[0][f] is None]
    return recs, {"missing": missing, "ai": ai_done}

