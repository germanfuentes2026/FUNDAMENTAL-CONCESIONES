"""Altman Z' / Z'' para empresas que no cotizan: X4 usa patrimonio contable."""
from __future__ import annotations

import math

FIELDS = [
    "current_assets", "current_liabilities", "total_assets", "total_liabilities",
    "equity", "retained_earnings", "ebit", "sales",
]
FACTOR_NAMES = [
    "X1 Capital de trabajo / Activo",
    "X2 Resultados acumulados / Activo",
    "X3 EBIT / Activo",
    "X4 Patrimonio neto / Pasivo",
    "X5 Ventas / Activo",
]
# w = pesos de X1..X5 (0 = el factor no entra), const = constante, lo/hi = cortes
MODELS = {
    "Z'' (no manufactureras)": dict(w=[6.56, 3.26, 6.72, 1.05, 0.0], const=0.0, lo=1.10, hi=2.60),
    "Z'' EM (mercados emergentes)": dict(w=[6.56, 3.26, 6.72, 1.05, 0.0], const=3.25, lo=4.15, hi=5.85),
    "Z' (privadas manufactureras)": dict(w=[0.717, 0.847, 3.107, 0.420, 0.998], const=0.0, lo=1.23, hi=2.90),
}


def _num(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


def _div(a, b):
    return None if a is None or b is None or b == 0 else a / b


def compute(rec: dict, model: str) -> dict:
    m = MODELS[model]
    g = {k: _num(rec.get(k)) for k in FIELDS}
    ca, cl = g["current_assets"], g["current_liabilities"]
    wc = None if ca is None or cl is None else ca - cl
    ta, tl = g["total_assets"], g["total_liabilities"]
    x = [_div(wc, ta), _div(g["retained_earnings"], ta), _div(g["ebit"], ta),
         _div(g["equity"], tl), _div(g["sales"], ta)]
    used = [i for i, w in enumerate(m["w"]) if w]
    missing = [f"X{i + 1}" for i in used if x[i] is None]
    contrib = [None if x[i] is None else m["w"][i] * x[i] for i in range(5)]
    z = None if missing else m["const"] + sum(contrib[i] for i in used)
    if z is None:
        zone = "n/a"
    elif z > m["hi"]:
        zone = "SAFE"
    elif z < m["lo"]:
        zone = "DISTRESS"
    else:
        zone = "GREY"
    return {"x": x, "contrib": contrib, "const": m["const"], "z": z, "zone": zone, "missing": missing}
