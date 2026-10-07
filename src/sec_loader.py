"""SEC EDGAR loader: ticker → CIK → companyfacts XBRL."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any

import requests

SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"

ANNUAL_FORMS = {"10-K", "10-K/A", "20-F", "20-F/A", "40-F", "40-F/A"}

CONCEPT_ALIASES: dict[str, list[str]] = {
    "assets": ["Assets"],
    "current_assets": ["AssetsCurrent"],
    "current_liabilities": ["LiabilitiesCurrent"],
    "liabilities": [
        "Liabilities",
    ],
    "long_term_debt": [
        "LongTermDebtNoncurrent",
        "LongTermDebt",
        "LongTermDebtAndCapitalLeaseObligations",
        "LongTermDebtAndCapitalLeaseObligationsNoncurrent",
        "LongTermDebtNoncurrentExcludingCapitalLeaseObligations",
    ],
    "net_income": ["NetIncomeLoss", "ProfitLoss", "NetIncomeLossAvailableToCommonStockholdersBasic"],
    "cfo": [
        "NetCashProvidedByUsedInOperatingActivities",
        "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
    ],
    "gross_profit": ["GrossProfit"],
    "revenue": [
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "SalesRevenueNet",
        "SalesRevenueGoodsNet",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
    ],
    "cost_of_revenue": [
        "CostOfRevenue",
        "CostOfGoodsAndServicesSold",
        "CostOfGoodsSold",
    ],
    "retained_earnings": [
        "RetainedEarningsAccumulatedDeficit",
        "RetainedEarningsUnappropriated",
    ],
    "ebit": [
        "OperatingIncomeLoss",
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments",
    ],
    "equity": [
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ],
    "shares": [
        "CommonStockSharesOutstanding",
        "WeightedAverageNumberOfDilutedSharesOutstanding",
        "WeightedAverageNumberOfSharesOutstandingBasic",
        "CommonStockSharesIssued",
        "EntityCommonStockSharesOutstanding",
    ],
}

USD_UNITS = {"USD", "USD/shares"}
SHARE_UNITS = {"shares", "pure"}


@dataclass
class FactPoint:
    concept: str
    value: float
    end: str
    fy: int | None
    fp: str | None
    form: str | None
    filed: str | None
    accn: str | None
    unit: str
    source: str = "us-gaap"


@dataclass
class CompanyMeta:
    ticker: str
    cik: str
    name: str
    sic: str | None = None
    sic_description: str | None = None
    exchanges: list[str] = field(default_factory=list)
    entity_type: str | None = None
    ein: str | None = None
    website: str | None = None


class SecLoaderError(RuntimeError):
    pass


def _user_agent() -> str:
    ua = os.environ.get("SEC_USER_AGENT", "").strip()
    if ua:
        return ua
    return "Fundamental Terminal academic research german.student@economicas.uba.ar"


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update(
        {
            "User-Agent": _user_agent(),
            "Accept-Encoding": "gzip, deflate",
            "Accept": "application/json, text/plain, */*",
        }
    )
    return s


def _get_json(session: requests.Session, url: str, *, retries: int = 4) -> Any:
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            resp = session.get(url, timeout=30)
            if resp.status_code == 429:
                time.sleep(1.5 * (attempt + 1))
                continue
            if resp.status_code == 404:
                raise SecLoaderError(f"SEC resource not found: {url}")
            resp.raise_for_status()
            return resp.json()
        except SecLoaderError:
            raise
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            time.sleep(0.6 * (attempt + 1))
    raise SecLoaderError(f"Failed to fetch {url}: {last_exc}")


def load_ticker_map(session: requests.Session | None = None) -> dict[str, dict[str, str]]:
    sess = session or _session()
    raw = _get_json(sess, SEC_TICKERS_URL)
    mapping: dict[str, dict[str, str]] = {}
    for row in raw.values():
        ticker = str(row.get("ticker", "")).upper().strip()
        if not ticker:
            continue
        cik = str(row.get("cik_str", "")).zfill(10)
        mapping[ticker] = {"cik": cik, "name": str(row.get("title", "")), "ticker": ticker}
    return mapping


def resolve_ticker(ticker: str, session: requests.Session | None = None) -> CompanyMeta:
    ticker_u = ticker.upper().strip()
    if not ticker_u:
        raise SecLoaderError("Enter a ticker symbol.")
    sess = session or _session()
    mapping = load_ticker_map(sess)
    hit = mapping.get(ticker_u)
    if not hit:
        raise SecLoaderError(f"Ticker {ticker_u} was not found in the SEC company tickers file.")
    return CompanyMeta(ticker=ticker_u, cik=hit["cik"], name=hit["name"])


def load_submissions(cik: str, session: requests.Session | None = None) -> dict[str, Any]:
    sess = session or _session()
    return _get_json(sess, SEC_SUBMISSIONS_URL.format(cik=cik.zfill(10)))


def enrich_meta(meta: CompanyMeta, submissions: dict[str, Any]) -> CompanyMeta:
    meta.name = submissions.get("name") or meta.name
    meta.sic = str(submissions.get("sic") or "") or None
    meta.sic_description = submissions.get("sicDescription")
    meta.exchanges = list(submissions.get("exchanges") or [])
    meta.entity_type = submissions.get("entityType")
    meta.ein = submissions.get("ein")
    meta.website = submissions.get("website")
    return meta


def load_company_facts(cik: str, session: requests.Session | None = None) -> dict[str, Any]:
    sess = session or _session()
    return _get_json(sess, SEC_FACTS_URL.format(cik=cik.zfill(10)))


def _iter_points(facts: dict[str, Any], concept: str) -> list[FactPoint]:
    points: list[FactPoint] = []
    for taxonomy in ("us-gaap", "ifrs-full", "dei"):
        node = (facts.get("facts") or {}).get(taxonomy, {}).get(concept)
        if not node:
            continue
        units = node.get("units") or {}
        for unit, rows in units.items():
            if unit not in USD_UNITS and unit not in SHARE_UNITS:
                if not unit.upper().startswith("USD"):
                    continue
            for row in rows:
                val = row.get("val")
                if val is None:
                    continue
                try:
                    value = float(val)
                except (TypeError, ValueError):
                    continue
                fy = row.get("fy")
                try:
                    fy_i = int(fy) if fy is not None else None
                except (TypeError, ValueError):
                    fy_i = None
                points.append(
                    FactPoint(
                        concept=concept,
                        value=value,
                        end=str(row.get("end") or ""),
                        fy=fy_i,
                        fp=row.get("fp"),
                        form=row.get("form"),
                        filed=row.get("filed"),
                        accn=row.get("accn"),
                        unit=unit,
                        source=taxonomy,
                    )
                )
    return points


def _prefer_annual(points: list[FactPoint]) -> list[FactPoint]:
    annual = [p for p in points if (p.form in ANNUAL_FORMS) and (p.fp in (None, "FY") or p.fp == "FY")]
    if annual:
        return annual
    fy_only = [p for p in points if p.fp == "FY"]
    return fy_only or points


def pick_annual_series(
    facts: dict[str, Any],
    aliases: list[str],
    *,
    prefer_shares: bool = False,
    max_years: int = 8,
) -> list[FactPoint]:
    """Latest annual observation per fiscal year, preferring 10-K / 20-F."""
    collected: list[FactPoint] = []
    for concept in aliases:
        collected.extend(_iter_points(facts, concept))
        if collected:
            # keep first concept family that yields data, but still allow later
            # aliases if the first only has sparse coverage
            pass

    if not collected:
        return []

    if prefer_shares:
        share_pts = [p for p in collected if p.unit in SHARE_UNITS]
        collected = share_pts or collected
    else:
        usd_pts = [p for p in collected if p.unit in USD_UNITS or p.unit.upper().startswith("USD")]
        collected = usd_pts or collected

    annual = _prefer_annual(collected)

    by_year: dict[int, FactPoint] = {}
    for p in annual:
        year = p.fy
        if year is None and p.end:
            try:
                year = int(p.end[:4])
            except ValueError:
                continue
        if year is None:
            continue
        prev = by_year.get(year)
        if prev is None:
            by_year[year] = p
            continue
        # Prefer exact concept order, then later filed, then later period end
        prev_rank = _alias_rank(prev.concept, aliases)
        new_rank = _alias_rank(p.concept, aliases)
        if new_rank < prev_rank:
            by_year[year] = p
            continue
        if new_rank == prev_rank:
            if (p.filed or "") > (prev.filed or "") or (
                (p.filed or "") == (prev.filed or "") and (p.end or "") > (prev.end or "")
            ):
                by_year[year] = p

    years = sorted(by_year.keys(), reverse=True)[:max_years]
    return [by_year[y] for y in sorted(years)]


def _alias_rank(concept: str, aliases: list[str]) -> int:
    try:
        return aliases.index(concept)
    except ValueError:
        return 999


def extract_statement_bundle(facts: dict[str, Any]) -> dict[str, list[FactPoint]]:
    bundle: dict[str, list[FactPoint]] = {}
    for key, aliases in CONCEPT_ALIASES.items():
        bundle[key] = pick_annual_series(facts, aliases, prefer_shares=(key == "shares"))
    if not bundle["gross_profit"] and bundle["revenue"] and bundle["cost_of_revenue"]:
        gp: list[FactPoint] = []
        costs = {p.fy: p for p in bundle["cost_of_revenue"] if p.fy is not None}
        for rev in bundle["revenue"]:
            if rev.fy is None or rev.fy not in costs:
                continue
            c = costs[rev.fy]
            gp.append(
                FactPoint(
                    concept="GrossProfit (derived: Revenue − COGS)",
                    value=rev.value - c.value,
                    end=rev.end,
                    fy=rev.fy,
                    fp=rev.fp,
                    form=rev.form,
                    filed=rev.filed,
                    accn=rev.accn,
                    unit=rev.unit,
                    source="derived",
                )
            )
        bundle["gross_profit"] = gp
    if not bundle["liabilities"] and bundle["assets"] and bundle["equity"]:
        eq = {p.fy: p for p in bundle["equity"] if p.fy is not None}
        derived: list[FactPoint] = []
        for a in bundle["assets"]:
            if a.fy is None or a.fy not in eq:
                continue
            e = eq[a.fy]
            derived.append(
                FactPoint(
                    concept="Liabilities (derived: Assets − Equity)",
                    value=a.value - e.value,
                    end=a.end,
                    fy=a.fy,
                    fp=a.fp,
                    form=a.form,
                    filed=a.filed,
                    accn=a.accn,
                    unit=a.unit,
                    source="derived",
                )
            )
        bundle["liabilities"] = derived
    return bundle


def series_by_year(points: list[FactPoint]) -> dict[int, FactPoint]:
    return {p.fy: p for p in points if p.fy is not None}


def latest_overlapping_years(*series: dict[int, FactPoint], need: int = 2) -> list[int]:
    years: set[int] | None = None
    for s in series:
        keys = set(s.keys())
        years = keys if years is None else years & keys
    if not years:
        all_years: set[int] = set()
        for s in series:
            all_years |= set(s.keys())
        return sorted(all_years, reverse=True)[:need]
    return sorted(years, reverse=True)[:need]


def fetch_market_snapshot(ticker: str) -> dict[str, Any]:
    """Last price / market snapshot. Yahoo first, Stooq fallback."""
    ticker_u = ticker.upper().strip()
    info: dict[str, Any] = {}
    info.update(_yahoo_chart_quote(ticker_u))
    if info.get("price") is None:
        info.update(_stooq_quote(ticker_u))
    if info.get("price") is None:
        info.update(_yfinance_quote(ticker_u))
    return {k: v for k, v in info.items() if v is not None}


def _yahoo_chart_quote(ticker: str) -> dict[str, Any]:
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
    try:
        resp = requests.get(
            url,
            params={"range": "5d", "interval": "1d"},
            headers={
                "User-Agent": "Mozilla/5.0",
                "Accept": "application/json",
            },
            timeout=20,
        )
        resp.raise_for_status()
        payload = resp.json()
        result = ((payload.get("chart") or {}).get("result") or [None])[0]
        if not result:
            return {}
        meta = result.get("meta") or {}
        closes = ((result.get("indicators") or {}).get("quote") or [{}])[0].get("close") or []
        price = _safe_float(meta.get("regularMarketPrice"))
        if price is None:
            for c in reversed(closes):
                price = _safe_float(c)
                if price is not None:
                    break
        shares = _safe_float(meta.get("sharesOutstanding"))
        mcap = None
        if price is not None and shares is not None:
            mcap = price * shares
        return {
            "price": price,
            "market_cap": mcap,
            "shares": shares,
            "currency": meta.get("currency"),
            "source": "yahoo-chart",
        }
    except Exception:  # noqa: BLE001
        return {}


def _stooq_quote(ticker: str) -> dict[str, Any]:
    url = "https://stooq.com/q/l/"
    try:
        resp = requests.get(
            url,
            params={"s": f"{ticker.lower()}.us", "f": "sd2t2ohlcv", "h": "", "e": "csv"},
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=20,
        )
        resp.raise_for_status()
        lines = [ln.strip() for ln in resp.text.splitlines() if ln.strip()]
        if len(lines) < 2:
            return {}
        parts = lines[1].split(",")
        # Symbol,Date,Time,Open,High,Low,Close,Volume
        close = _safe_float(parts[6] if len(parts) > 6 else None)
        if close is None:
            return {}
        return {"price": close, "source": "stooq"}
    except Exception:  # noqa: BLE001
        return {}


def _yfinance_quote(ticker: str) -> dict[str, Any]:
    try:
        import yfinance as yf  # type: ignore
    except ImportError:
        return {}
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period="5d")
        price = None
        if hist is not None and not hist.empty:
            price = _safe_float(hist["Close"].dropna().iloc[-1])
        return {"price": price, "source": "yfinance"} if price is not None else {}
    except Exception:  # noqa: BLE001
        return {}


def _safe_float(v: Any) -> float | None:
    try:
        if v is None:
            return None
        return float(v)
    except (TypeError, ValueError):
        return None
