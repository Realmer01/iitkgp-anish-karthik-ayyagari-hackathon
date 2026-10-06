"""Generate the synthetic wholesale-banking trade blotter for Module B (seeded -> identical every run).

    python scripts/generate_portfolio.py

The case-study pack did not include sample transaction data, and the transaction datasets it points to
(Salad Money open banking, Kaggle financial transactions) are retail/card data, not wholesale. So we
generate a wholesale blotter with documented assumptions:
  * LOANS go to FICTIONAL borrowers (private lending relationships are never public data);
  * BONDS, CDS and EQUITY reference PUBLIC issuers (index constituents, sovereigns, CDX indices) -
    ordinary trading-book holdings, which lets engine company signals map onto real exposures;
  * dealers on derivatives are fictional; nothing relates to any real bank's book or client.
Outputs (data/portfolio/): counterparties.csv, transactions.csv (blotter), rating_params.csv, market_params.csv
"""
from __future__ import annotations

from datetime import date, timedelta

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

from riskpulse.config import PORTFOLIO, PRICES

RNG = np.random.default_rng(20261006)
AS_OF = date(2026, 10, 6)

RATING_PARAMS = pd.DataFrame({  # spreads: typical USD OAS; PD: long-run average 1y default rates (approx.)
    "rating": ["AAA", "AA", "A", "BBB", "BB", "B", "CCC"],
    "spread_bp": [35, 55, 85, 135, 260, 420, 900],
    "pd_1y": [0.0001, 0.0002, 0.0006, 0.0018, 0.0070, 0.0300, 0.2500],
    "rw_corporate": [0.20, 0.20, 0.50, 0.75, 1.00, 1.50, 1.50],    # Basel III standardised (approx.)
    "rw_sovereign": [0.00, 0.00, 0.20, 0.50, 1.00, 1.00, 1.50],
})
CCY = {"USD": (None, 1.0), "EUR": (0.0265, 1.09), "GBP": (0.0420, 1.31), "JPY": (0.0160, 0.0068),
       "BRL": (0.1350, 0.18), "INR": (0.0650, 0.012), "CNH": (0.0190, 0.14)}  # base rate, USD per unit
REGION = {  # region -> (currency, countries, share of fictional borrowers)
    "North America": ("USD", ["US", "US", "US", "CA", "MX"], 0.46), "Europe": ("EUR", ["DE", "FR", "GB", "IT", "ES", "NL"], 0.24),
    "Asia-Pacific": ("JPY", ["JP", "KR", "IN", "AU", "SG"], 0.10), "China": ("CNH", ["CN"], 0.06),
    "Middle East": ("USD", ["AE", "SA", "IL", "QA"], 0.06), "Latin America": ("BRL", ["BR", "CL", "CO"], 0.06),
    "Russia/CIS": ("USD", ["KZ"], 0.02),
}
SECTORS = ["Energy", "Financials", "Industrials", "Information Technology", "Health Care", "Consumer Staples",
           "Consumer Discretionary", "Communication Services", "Real Estate", "Utilities", "Materials"]
SUFFIX = dict(zip(SECTORS, ["Energy", "Capital", "Industries", "Systems", "Health", "Foods", "Retail", "Media",
                            "Properties", "Power", "Materials"]))
PREFIX = ["Northwind", "Halcyon", "Brightwater", "Cedar", "Orion", "Ashford", "Pioneer", "Vantage", "Lumen", "Granite",
          "Kestrel", "Sable", "Atlas", "Meridian", "Harbor", "Quarry", "Zenith", "Evergreen", "Tidewater", "Blue Ridge",
          "Silverline", "Redwood", "Summit", "Ironclad", "Coral", "Falcon", "Juniper", "Beacon", "Crescent", "Aurora",
          "Oakmont", "Riverstone", "Solano", "Amberly", "Calder", "Driftwood", "Elmstead", "Foxglove", "Glenmore",
          "Hawthorn", "Ivory", "Jasper", "Kingsley", "Larkspur", "Marlow", "Norfield", "Osprey", "Pembrook"]
PUBLIC = [("AAPL", "Apple Inc.", "Information Technology", "AA"), ("MSFT", "Microsoft Corp.", "Information Technology", "AAA"),
          ("AMZN", "Amazon.com Inc.", "Consumer Discretionary", "AA"), ("GOOGL", "Alphabet Inc.", "Communication Services", "AA"),
          ("META", "Meta Platforms Inc.", "Communication Services", "AA"), ("NVDA", "NVIDIA Corp.", "Information Technology", "AA"),
          ("JPM", "JPMorgan Chase & Co.", "Financials", "A"), ("GS", "Goldman Sachs Group Inc.", "Financials", "BBB"),
          ("XOM", "Exxon Mobil Corp.", "Energy", "AA"), ("JNJ", "Johnson & Johnson", "Health Care", "AAA"),
          ("PFE", "Pfizer Inc.", "Health Care", "A"), ("BA", "Boeing Co.", "Industrials", "BBB"),
          ("CAT", "Caterpillar Inc.", "Industrials", "A"), ("WMT", "Walmart Inc.", "Consumer Staples", "AA"),
          ("KO", "Coca-Cola Co.", "Consumer Staples", "A"), ("TSLA", "Tesla Inc.", "Consumer Discretionary", "BBB")]
SOVEREIGN = [("UST", "US Treasury", "North America", "US", "USD", "AA"), ("BUND", "Germany Bund", "Europe", "DE", "EUR", "AAA"),
             ("GILT", "UK Gilt", "Europe", "GB", "GBP", "AA"), ("JGB", "Japan JGB", "Asia-Pacific", "JP", "JPY", "A"),
             ("BTP", "Italy BTP", "Europe", "IT", "EUR", "BBB"), ("BRAZIL", "Brazil NTN-F", "Latin America", "BR", "BRL", "BB"),
             ("INDIA", "India G-Sec", "Asia-Pacific", "IN", "INR", "BBB")]
DEALERS = ["Silverline Bank", "Monarch Securities", "Polaris Bank", "Westgate Markets", "Sterling Bank"]


def _rating() -> str:
    if RNG.random() < 0.68:
        return str(RNG.choice(["AA", "A", "BBB", "BBB"], p=[0.08, 0.32, 0.32, 0.28]))
    return str(RNG.choice(["BB", "B", "CCC"], p=[0.55, 0.38, 0.07]))


def counterparties() -> pd.DataFrame:
    rows, names = [], [n for p in PREFIX for n in (p, f"{p} Group")]
    regions = list(REGION)
    for i in range(90):
        sector, region = SECTORS[i % len(SECTORS)], str(RNG.choice(regions, p=[REGION[r][2] for r in regions]))
        name = f"{names[i % len(names)].strip()} {SUFFIX[sector]}"
        rows.append(dict(cpty_id=f"C{i + 1:03d}", name=name, type="Corporate (fictional)", sector=sector, region=region,
                         country=str(RNG.choice(REGION[region][1])), currency=REGION[region][0], rating=_rating(), ticker=""))
    for t, name, sector, rating in PUBLIC:
        rows.append(dict(cpty_id=f"P-{t}", name=name, type="Public issuer", sector=sector, region="North America",
                         country="US", currency="USD", rating=rating, ticker=t))
    for code, name, region, country, ccy, rating in SOVEREIGN:
        rows.append(dict(cpty_id=f"S-{code}", name=name, type="Sovereign", sector="Sovereign", region=region,
                         country=country, currency=ccy, rating=rating, ticker=""))
    for i, name in enumerate(DEALERS):
        rows.append(dict(cpty_id=f"D{i + 1:02d}", name=f"{name} (fictional dealer)", type="Dealer", sector="Financials",
                         region="Europe" if i % 2 else "North America", country="GB" if i % 2 else "US",
                         currency="USD", rating=str(RNG.choice(["AA", "A", "A"])), ticker=""))
    for code, name, rating in (("CDX-IG", "CDX North America IG index", "BBB"), ("CDX-HY", "CDX North America HY index", "B")):
        rows.append(dict(cpty_id=f"I-{code}", name=name, type="Index", sector="Diversified", region="North America",
                         country="US", currency="USD", rating=rating, ticker=""))
    return pd.DataFrame(rows)


def blotter(cp: pd.DataFrame) -> pd.DataFrame:
    tx = []
    spread = RATING_PARAMS.set_index("rating")["spread_bp"]
    dealers = [f"D{j + 1:02d}" for j in range(len(DEALERS))]
    usd_rate = float(pd.read_csv(PRICES / "prices_daily.csv", index_col=0)["^TNX"].dropna().iloc[-1]) / 100
    rate = {c: (r if r is not None else usd_rate) for c, (r, _) in CCY.items()}

    def add(position_id, cpty_id, instrument, asset_class, side, notional, currency, maturity="", rate_=0.0,
            seniority="", price=100.0, underlying="", days_back=900):
        tx.append(dict(trade_date=(AS_OF - timedelta(days=int(RNG.integers(5, days_back)))).isoformat(),
                       position_id=position_id, cpty_id=cpty_id, instrument=instrument, asset_class=asset_class,
                       side=side, notional=float(round(notional, -3)), currency=currency, maturity=maturity,
                       coupon_or_rate=round(float(rate_), 5), seniority=seniority, price=round(float(price), 4),
                       underlying=underlying))

    def mat(lo, hi):
        return (AS_OF + timedelta(days=int(365 * RNG.uniform(lo, hi)))).isoformat()

    for c in cp[cp.type == "Corporate (fictional)"].itertuples():           # ---- loan book
        size = float(RNG.lognormal(np.log(55e6 if c.rating in ("AA", "A", "BBB") else 35e6), 0.45))
        products = ["Term Loan"] + (["Revolving Credit Facility"] if RNG.random() < 0.55 else []) \
            + (["Trade Finance"] if c.region != "North America" and RNG.random() < 0.45 else [])
        for prod in products:
            pid = f"LN-{c.cpty_id}-{prod.split()[0][:3].upper()}"
            secured = "Senior Secured" if c.rating in ("BB", "B", "CCC") or RNG.random() < 0.3 else "Senior Unsecured"
            m = mat(0.3, 1.5) if prod == "Trade Finance" else mat(1.5, 7)
            margin = spread[c.rating] * RNG.uniform(0.9, 1.3) / 1e4
            commit = size * (1.6 if prod == "Revolving Credit Facility" else 1.0)
            add(pid, c.cpty_id, prod, "Loan", "Commitment", commit, c.currency, m, margin, secured)
            drawn = 0.0
            for _ in range(int(RNG.integers(1, 4))):
                amt = min(commit - drawn, commit * RNG.uniform(0.25, 0.6))
                if amt <= 1e5:
                    break
                drawn += amt
                add(pid, c.cpty_id, prod, "Loan", "Drawdown", amt, c.currency, m, margin, secured)
            if drawn and RNG.random() < 0.3:
                add(pid, c.cpty_id, prod, "Loan", "Repayment", -drawn * RNG.uniform(0.1, 0.3), c.currency, m, margin,
                    secured, days_back=200)

    issuers = list(cp[cp.type.isin(["Sovereign", "Public issuer"])].itertuples()) + \
        list(cp[(cp.type == "Corporate (fictional)") & cp.rating.isin(["BB", "B"])].sample(12, random_state=1).itertuples())
    for c in issuers:                                                       # ---- bond book
        size = 260e6 if c.type == "Sovereign" else 60e6
        cpn = max(0.005, rate[c.currency] + spread.get(c.rating, 100) / 1e4 * (0.25 if c.type == "Sovereign" else 1)
                  + RNG.normal(0, 0.003))
        m, inst = mat(2, 12), "Sovereign Bond" if c.type == "Sovereign" else "Corporate Bond"
        for side, frac in (("Buy", 1.0), ("Buy", RNG.uniform(0.2, 0.5)), ("Sell", -RNG.uniform(0.1, 0.4))):
            add(f"BD-{c.cpty_id}", c.cpty_id, inst, "Bond", side, size * frac * RNG.uniform(0.6, 1.4), c.currency, m,
                cpn, "Senior Unsecured", price=RNG.uniform(95, 103))

    for i in range(16):                                                     # ---- rates hedges
        ccy = str(RNG.choice(["USD", "USD", "USD", "EUR", "GBP", "JPY"]))
        add(f"IRS-{i + 1:02d}", str(RNG.choice(dealers)), "Interest Rate Swap", "Derivative",
            "Pay Fixed" if RNG.random() < 0.7 else "Receive Fixed", RNG.uniform(100e6, 400e6), ccy, mat(2, 10),
            rate[ccy] + RNG.normal(0, 0.003))
    for t in ["BA", "GS", "TSLA", "PFE", "JPM", "XOM"]:                      # ---- credit hedges
        add(f"CDS-{t}", str(RNG.choice(dealers)), "Credit Default Swap", "Derivative", "Buy Protection",
            RNG.uniform(30e6, 90e6), "USD", mat(3, 5), 0.01, "Senior Unsecured", underlying=f"P-{t}")
    for idx, ntl, cpn in (("CDX-IG", 300e6, 0.01), ("CDX-HY", 80e6, 0.05)):
        add(f"CDS-{idx}", "D01", "CDS Index", "Derivative", "Buy Protection", ntl, "USD", mat(4.5, 5), cpn,
            underlying=f"I-{idx}")
    for ccy, ntl in (("EUR", 700e6), ("GBP", 160e6), ("JPY", 220e6), ("BRL", 90e6), ("CNH", 110e6), ("INR", 60e6)):
        add(f"FXF-{ccy}USD", str(RNG.choice(dealers)), "FX Forward", "Derivative", "Sell Foreign",
            ntl * RNG.uniform(0.7, 1.0), ccy, mat(0.2, 1), days_back=300)
    px = pd.read_csv(PRICES / "prices_daily.csv", index_col=0).ffill().iloc[-1]
    for t, *_ in PUBLIC:                                                    # ---- equity book
        add(f"EQ-{t}", f"P-{t}", "Listed Equity", "Equity", "Buy", round(RNG.uniform(12e6, 45e6) / px[t]), "USD",
            price=px[t], underlying=t, days_back=150)
    for t in ["NVDA", "AMZN", "META", "XOM"]:                               # client-facilitation total-return swaps
        add(f"TRS-{t}", str(RNG.choice(dealers)), "Equity Total Return Swap", "Derivative", "Receive Equity",
            RNG.uniform(20e6, 50e6), "USD", mat(0.5, 1.5), underlying=t)
    for k, (strike, ntl) in enumerate([(0.95, 200e6), (0.90, 150e6), (0.85, 100e6)]):
        add(f"OPT-SPX-P{k + 1}", "D02", "Equity Index Put Option", "Derivative", "Buy", ntl, "USD", mat(0.25, 0.75),
            strike, underlying="SPY")
    add("EQF-SPX", "D02", "Equity Index Future", "Derivative", "Sell", 180e6, "USD", mat(0.2, 0.25), underlying="SPY")
    add("CMD-WTI", "D03", "Commodity Swap", "Derivative", "Pay Floating", 120e6, "USD", mat(0.5, 1.5), underlying="CL=F")

    df = pd.DataFrame(tx).merge(cp[["cpty_id", "name", "sector", "country", "region", "rating"]], on="cpty_id")
    df = df.rename(columns={"name": "counterparty"}).sort_values(["trade_date", "position_id"]).reset_index(drop=True)
    df.insert(0, "trade_id", [f"T{i + 1:05d}" for i in range(len(df))])
    return df


def main() -> None:
    PORTFOLIO.mkdir(parents=True, exist_ok=True)
    cp = counterparties()
    tx = blotter(cp)
    usd = float(pd.read_csv(PRICES / "prices_daily.csv", index_col=0)["^TNX"].dropna().iloc[-1]) / 100
    mkt = pd.DataFrame([{"currency": c, "base_rate": round(r if r is not None else usd, 4), "usd_per_unit": fx}
                        for c, (r, fx) in CCY.items()])
    cp.to_csv(PORTFOLIO / "counterparties.csv", index=False)
    tx.to_csv(PORTFOLIO / "transactions.csv", index=False)
    RATING_PARAMS.to_csv(PORTFOLIO / "rating_params.csv", index=False)
    mkt.to_csv(PORTFOLIO / "market_params.csv", index=False)
    print(f"{len(cp)} counterparties, {len(tx)} trades, {tx.position_id.nunique()} positions; USD rate {usd:.2%}")
    print(tx.groupby("asset_class").position_id.nunique().to_dict())


if __name__ == "__main__":
    main()
