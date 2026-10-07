"""Extrae los datos de Altman de un balance en PDF (escaneado o con texto) usando la API de Claude."""
from __future__ import annotations

import base64
import io
import json
import os
import re

MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5")
MAX_PAGES = 90  # la API procesa hasta ~100 páginas por PDF
KEYWORDS = re.compile(
    r"estado de situaci[oó]n|balance general|estado de resultados|estado del resultado|"
    r"evoluci[oó]n del patrimonio|total del activo|total activo", re.I)

PROMPT = """Sos un analista contable argentino. Del PDF adjunto (estados contables de una concesionaria vial)
extraé los datos para calcular el Altman Z-Score. Devolvé SOLO un JSON, sin texto ni markdown, con esta forma:

{"company": "razón social",
 "periods": [
  {"period_end": "YYYY-MM-DD", "column": "current" | "prior",
   "current_assets": n, "current_liabilities": n, "total_assets": n, "total_liabilities": n,
   "equity": n, "retained_earnings": n, "ebit": n, "sales": n, "notes": "texto breve"}
 ]}

Reglas:
- Una entrada por cada columna del estado de situación patrimonial: el ejercicio actual ("current") y el comparativo ("prior").
- Números en unidades completas (si el estado dice "en millones", multiplicá por 1.000.000). Pérdidas con signo negativo.
- retained_earnings = reservas (legal, facultativa, otras) + resultados no asignados/acumulados (incluye el resultado del ejercicio). NO incluyas capital social ni ajuste de capital.
- ebit = resultado operativo antes de resultados financieros (intereses, diferencias de cambio, RECPAM, tenencia) y antes de impuesto a las ganancias.
  Si no hay esa línea, calculalo como resultado antes de impuesto menos resultados financieros netos. En "notes" aclará cómo lo obtuviste.
- sales = ingresos operativos (peajes / concesión).
- Si un dato no figura, usá null. No inventes valores."""


def select_pages(pdf_bytes: bytes) -> bytes:
    """Si el PDF es muy largo, deja solo las páginas de estados contables."""
    from pypdf import PdfReader, PdfWriter

    reader = PdfReader(io.BytesIO(pdf_bytes))
    n = len(reader.pages)
    if n <= MAX_PAGES:
        return pdf_bytes
    hits = []
    for i, p in enumerate(reader.pages):
        try:
            if KEYWORDS.search(p.extract_text() or ""):
                hits.append(i)
        except Exception:  # noqa: BLE001
            continue
    keep = sorted({j for i in hits for j in (i, i + 1) if j < n})[:MAX_PAGES] or list(range(MAX_PAGES))
    w = PdfWriter()
    for i in keep:
        w.add_page(reader.pages[i])
    buf = io.BytesIO()
    w.write(buf)
    return buf.getvalue()


def parse_json(text: str) -> dict:
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    a, b = text.find("{"), text.rfind("}")
    if a < 0 or b < 0:
        raise ValueError("La respuesta no contiene un JSON.")
    return json.loads(text[a:b + 1])


def normalize(result: dict, source: str, company: str | None = None) -> list[dict]:
    recs = []
    for p in result.get("periods", []):
        end = str(p.get("period_end") or "")[:10]
        if not re.match(r"\d{4}-\d{2}-\d{2}$", end):
            continue
        recs.append({
            "company": company or result.get("company") or "Sin nombre",
            "fy": int(end[:4]), "period_end": end,
            "origin": "own" if p.get("column") == "current" else "comparative",
            "source": source,
            **{k: p.get(k) for k in ("current_assets", "current_liabilities", "total_assets",
                                      "total_liabilities", "equity", "retained_earnings", "ebit", "sales")},
            "notes": p.get("notes") or "",
        })
    return recs


def extract(pdf_bytes: bytes, api_key: str, source: str, company: str | None = None) -> list[dict]:
    import anthropic

    data = base64.standard_b64encode(select_pages(pdf_bytes)).decode()
    client = anthropic.Anthropic(api_key=api_key)
    msg = client.messages.create(
        model=MODEL, max_tokens=3000,
        messages=[{"role": "user", "content": [
            {"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": data}},
            {"type": "text", "text": PROMPT}]}],
    )
    text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
    return normalize(parse_json(text), source, company)
