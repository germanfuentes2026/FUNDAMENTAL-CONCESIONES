"""Piotroski F-Score (2000) with formula-level audit."""

from __future__ import annotations

from dataclasses import dataclass, field

from src.sec_loader import FactPoint, series_by_year


@dataclass
class CriterionResult:
    code: str
    name: str
    group: str
    passed: bool | None
    points: int
    formula: str
    latex: str
    explanation: str
    inputs: dict[str, float | int | str | None] = field(default_factory=dict)
    narrative: str = ""


@dataclass
class PiotroskiResult:
    score: int | None
    max_score: int = 9
    year: int | None = None
    prior_year: int | None = None
    criteria: list[CriterionResult] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        if self.score is None:
            return "INCOMPLETE"
        if self.score >= 8:
            return "HIGH QUALITY"
        if self.score >= 5:
            return "AVERAGE"
        return "LOW QUALITY"


def _v(series: dict[int, FactPoint], year: int | None) -> float | None:
    if year is None:
        return None
    p = series.get(year)
    return None if p is None else p.value


def _pt(series: dict[int, FactPoint], year: int | None) -> FactPoint | None:
    if year is None:
        return None
    return series.get(year)


def _ratio(num: float | None, den: float | None) -> float | None:
    if num is None or den is None or den == 0:
        return None
    return num / den


def _flag(cond: bool | None) -> tuple[bool | None, int]:
    if cond is None:
        return None, 0
    return cond, int(bool(cond))


def compute_piotroski(bundle: dict[str, list[FactPoint]]) -> PiotroskiResult:
    assets = series_by_year(bundle.get("assets") or [])
    ni = series_by_year(bundle.get("net_income") or [])
    cfo = series_by_year(bundle.get("cfo") or [])
    ltd = series_by_year(bundle.get("long_term_debt") or [])
    ca = series_by_year(bundle.get("current_assets") or [])
    cl = series_by_year(bundle.get("current_liabilities") or [])
    shares = series_by_year(bundle.get("shares") or [])
    gp = series_by_year(bundle.get("gross_profit") or [])
    rev = series_by_year(bundle.get("revenue") or [])

    years = sorted(set(assets) & set(ni), reverse=True)
    if len(years) < 2:
        years = sorted(set(assets.keys()) | set(ni.keys()), reverse=True)
    if not years:
        return PiotroskiResult(score=None, missing=["No overlapping annual XBRL years for assets/net income."])

    y = years[0]
    y1 = years[1] if len(years) > 1 else None

    a_t, a_p = _v(assets, y), _v(assets, y1)
    ni_t, ni_p = _v(ni, y), _v(ni, y1)
    cfo_t = _v(cfo, y)
    ltd_t, ltd_p = _v(ltd, y), _v(ltd, y1)
    ca_t, ca_p = _v(ca, y), _v(ca, y1)
    cl_t, cl_p = _v(cl, y), _v(cl, y1)
    sh_t, sh_p = _v(shares, y), _v(shares, y1)
    gp_t, gp_p = _v(gp, y), _v(gp, y1)
    rev_t, rev_p = _v(rev, y), _v(rev, y1)

    avg_a_t = None if a_t is None or a_p is None else (a_t + a_p) / 2
    avg_a_p = None
    if y1 is not None:
        years_sorted = sorted(assets.keys())
        idx = years_sorted.index(y1) if y1 in years_sorted else -1
        a_pp = assets[years_sorted[idx - 1]].value if idx > 0 else None
        if a_p is not None and a_pp is not None:
            avg_a_p = (a_p + a_pp) / 2

    roa_t = _ratio(ni_t, avg_a_t if avg_a_t else a_t)
    roa_p = _ratio(ni_p, avg_a_p if avg_a_p else a_p)
    d_roa = None if roa_t is None or roa_p is None else roa_t - roa_p

    lev_t = _ratio(ltd_t, a_t)
    lev_p = _ratio(ltd_p, a_p)
    d_lev = None if lev_t is None or lev_p is None else lev_t - lev_p

    cr_t = _ratio(ca_t, cl_t)
    cr_p = _ratio(ca_p, cl_p)
    d_cr = None if cr_t is None or cr_p is None else cr_t - cr_p

    gm_t = _ratio(gp_t, rev_t)
    gm_p = _ratio(gp_p, rev_p)
    d_gm = None if gm_t is None or gm_p is None else gm_t - gm_p

    at_t = _ratio(rev_t, avg_a_t if avg_a_t else a_t)
    at_p = _ratio(rev_p, avg_a_p if avg_a_p else a_p)
    d_at = None if at_t is None or at_p is None else at_t - at_p

    f_roa, p_roa = _flag(None if roa_t is None else roa_t > 0)
    f_cfo, p_cfo = _flag(None if cfo_t is None else cfo_t > 0)
    f_droa, p_droa = _flag(None if d_roa is None else d_roa > 0)
    accr = None if cfo_t is None or ni_t is None else cfo_t > ni_t
    f_acc, p_acc = _flag(accr)
    f_lev, p_lev = _flag(None if d_lev is None else d_lev < 0)
    f_liq, p_liq = _flag(None if d_cr is None else d_cr > 0)
    # Dilution: F_EQ = 1 if shares outstanding did not increase
    eq_ok = None if sh_t is None or sh_p is None else sh_t <= sh_p * 1.002
    f_eq, p_eq = _flag(eq_ok)
    f_gm, p_gm = _flag(None if d_gm is None else d_gm > 0)
    f_at, p_at = _flag(None if d_at is None else d_at > 0)

    def src(series: dict[int, FactPoint], year: int | None) -> dict[str, float | int | str | None]:
        p = _pt(series, year)
        if not p:
            return {"year": year, "value": None, "concept": None, "end": None, "form": None, "accn": None}
        return {
            "year": p.fy,
            "value": p.value,
            "concept": p.concept,
            "end": p.end,
            "form": p.form,
            "accn": p.accn,
            "unit": p.unit,
            "source": p.source,
        }

    criteria = [
        CriterionResult(
            code="F_ROA",
            name="Positive ROA",
            group="Profitability",
            passed=f_roa,
            points=p_roa,
            formula="ROA_t = NI_t / Avg(Assets); F_ROA = 1 if ROA_t > 0",
            latex=r"F_{ROA}=1 \iff \frac{NI_t}{\overline{A}_t}>0",
            explanation="Net income must be positive relative to assets (Piotroski uses beginning assets; we use average assets when both years exist).",
            inputs={
                "NI_t": ni_t,
                "Assets_t": a_t,
                "Assets_{t-1}": a_p,
                "ROA_t": roa_t,
                **{f"NI_src.{k}": v for k, v in src(ni, y).items()},
                **{f"Assets_src.{k}": v for k, v in src(assets, y).items()},
            },
            narrative=f"ROA_t = {roa_t:.4%}" if roa_t is not None else "ROA unavailable",
        ),
        CriterionResult(
            code="F_CFO",
            name="Positive operating cash flow",
            group="Profitability",
            passed=f_cfo,
            points=p_cfo,
            formula="F_CFO = 1 if CFO_t > 0",
            latex=r"F_{CFO}=1 \iff CFO_t>0",
            explanation="Cash from operations should be positive in the current year.",
            inputs={"CFO_t": cfo_t, **{f"CFO_src.{k}": v for k, v in src(cfo, y).items()}},
            narrative=f"CFO_t = {cfo_t:,.0f}" if cfo_t is not None else "CFO unavailable",
        ),
        CriterionResult(
            code="F_ΔROA",
            name="ROA increased",
            group="Profitability",
            passed=f_droa,
            points=p_droa,
            formula="F_ΔROA = 1 if ROA_t − ROA_{t-1} > 0",
            latex=r"F_{\Delta ROA}=1 \iff ROA_t-ROA_{t-1}>0",
            explanation="Year-over-year improvement in return on assets.",
            inputs={"ROA_t": roa_t, "ROA_{t-1}": roa_p, "ΔROA": d_roa, "NI_{t-1}": ni_p},
            narrative=f"ΔROA = {d_roa:.4%}" if d_roa is not None else "ΔROA unavailable",
        ),
        CriterionResult(
            code="F_ACCRUAL",
            name="Cash flow exceeds net income",
            group="Profitability",
            passed=f_acc,
            points=p_acc,
            formula="F_ACCRUAL = 1 if CFO_t > NI_t",
            latex=r"F_{ACCRUAL}=1 \iff CFO_t>NI_t",
            explanation="Quality-of-earnings screen: operating cash should exceed accruals-heavy earnings.",
            inputs={"CFO_t": cfo_t, "NI_t": ni_t, "CFO_t − NI_t": None if cfo_t is None or ni_t is None else cfo_t - ni_t},
            narrative="CFO > NI" if f_acc else ("CFO ≤ NI" if f_acc is False else "Accrual test unavailable"),
        ),
        CriterionResult(
            code="F_ΔLEVER",
            name="Leverage decreased",
            group="Leverage / liquidity / dilution",
            passed=f_lev,
            points=p_lev,
            formula="Leverage = LTD / Assets; F_ΔLEVER = 1 if ΔLeverage < 0",
            latex=r"F_{\Delta LEVER}=1 \iff \frac{LTD_t}{A_t}-\frac{LTD_{t-1}}{A_{t-1}}<0",
            explanation="Long-term debt ratio should fall (deleveraging).",
            inputs={"LTD_t": ltd_t, "LTD_{t-1}": ltd_p, "Lev_t": lev_t, "Lev_{t-1}": lev_p, "ΔLev": d_lev},
            narrative=f"ΔLeverage = {d_lev:.4%}" if d_lev is not None else "Leverage change unavailable",
        ),
        CriterionResult(
            code="F_ΔLIQUID",
            name="Current ratio increased",
            group="Leverage / liquidity / dilution",
            passed=f_liq,
            points=p_liq,
            formula="CR = CA / CL; F_ΔLIQUID = 1 if ΔCR > 0",
            latex=r"F_{\Delta LIQUID}=1 \iff \frac{CA_t}{CL_t}-\frac{CA_{t-1}}{CL_{t-1}}>0",
            explanation="Liquidity improvement via current ratio.",
            inputs={"CA_t": ca_t, "CL_t": cl_t, "CR_t": cr_t, "CR_{t-1}": cr_p, "ΔCR": d_cr},
            narrative=f"ΔCurrent ratio = {d_cr:.4f}" if d_cr is not None else "Current ratio change unavailable",
        ),
        CriterionResult(
            code="F_EQ",
            name="No share dilution",
            group="Leverage / liquidity / dilution",
            passed=f_eq,
            points=p_eq,
            formula="F_EQ = 1 if Shares_t ≤ Shares_{t-1}",
            latex=r"F_{EQ}=1 \iff Shares_t \le Shares_{t-1}",
            explanation="No equity issuance (shares outstanding did not increase; 0.2% tolerance for rounding).",
            inputs={"Shares_t": sh_t, "Shares_{t-1}": sh_p},
            narrative=(
                f"Shares {sh_t:,.0f} vs prior {sh_p:,.0f}"
                if sh_t is not None and sh_p is not None
                else "Shares outstanding unavailable"
            ),
        ),
        CriterionResult(
            code="F_ΔMARGIN",
            name="Gross margin increased",
            group="Operating efficiency",
            passed=f_gm,
            points=p_gm,
            formula="GM = GrossProfit / Revenue; F_ΔMARGIN = 1 if ΔGM > 0",
            latex=r"F_{\Delta MARGIN}=1 \iff \frac{GP_t}{S_t}-\frac{GP_{t-1}}{S_{t-1}}>0",
            explanation="Gross margin expansion as a proxy for pricing power / cost control.",
            inputs={"GP_t": gp_t, "Rev_t": rev_t, "GM_t": gm_t, "GM_{t-1}": gm_p, "ΔGM": d_gm},
            narrative=f"ΔGross margin = {d_gm:.4%}" if d_gm is not None else "Gross margin change unavailable",
        ),
        CriterionResult(
            code="F_ΔTURN",
            name="Asset turnover increased",
            group="Operating efficiency",
            passed=f_at,
            points=p_at,
            formula="ATO = Revenue / Avg(Assets); F_ΔTURN = 1 if ΔATO > 0",
            latex=r"F_{\Delta TURN}=1 \iff \frac{S_t}{\overline{A}_t}-\frac{S_{t-1}}{\overline{A}_{t-1}}>0",
            explanation="Productivity of the asset base improved.",
            inputs={"Rev_t": rev_t, "Rev_{t-1}": rev_p, "ATO_t": at_t, "ATO_{t-1}": at_p, "ΔATO": d_at},
            narrative=f"ΔAsset turnover = {d_at:.4f}" if d_at is not None else "Asset turnover change unavailable",
        ),
    ]

    missing: list[str] = []
    for c in criteria:
        if c.passed is None:
            missing.append(c.code)

    scored = [c.points for c in criteria if c.passed is not None]
    score = sum(c.points for c in criteria) if scored else None
    if score is not None and missing:
        # still report partial score
        pass

    return PiotroskiResult(
        score=0 if score is None and not missing else (sum(c.points for c in criteria) if any(c.passed is not None for c in criteria) else None),
        year=y,
        prior_year=y1,
        criteria=criteria,
        missing=missing,
    )
