"""Domain lexicons shared by the NLP components.

They have four jobs:
  1. weak-supervision labelling functions that map the public 20-topic tweet corpus onto
     RiskPulse's risk-oriented event taxonomy (scripts/build_training_data.py);
  2. high-precision rule overrides (e.g. "files for Chapter 11" is always a Credit Event);
  3. severity cues for the impact score;
  4. region / sector / market-wide linking.

All patterns are compiled case-insensitive with word boundaries.
"""
from __future__ import annotations

import re
from functools import lru_cache


def _rx(terms: list[str]) -> re.Pattern:
    body = "|".join(sorted((re.escape(t).replace(r"\ ", r"\s+") for t in terms), key=len, reverse=True))
    return re.compile(rf"(?<![\w$])(?:{body})(?!\w)", re.IGNORECASE)


# ---------------------------------------------------------------- event cue words
GEOPOLITICAL = [
    "war", "warfare", "invasion", "invade", "invades", "military", "missile", "missiles", "airstrike",
    "airstrikes", "troops", "sanction", "sanctions", "embargo", "tariff", "tariffs", "trade war",
    "geopolitical", "coup", "ceasefire", "nato", "conflict", "border clash", "annexation",
    "election", "elections", "parliament", "prime minister", "president", "white house", "kremlin",
    "beijing", "taiwan strait", "south china sea", "middle east", "terror", "terrorist", "nuclear",
    "export controls", "export ban", "retaliation", "diplomatic", "geopolitics", "regime", "protests",
    "unrest", "hostilities", "drone strike", "blockade", "red sea", "strait of hormuz",
]
CREDIT = [
    "default", "defaults", "defaulted", "bankruptcy", "bankrupt", "chapter 11", "chapter 7",
    "insolvency", "insolvent", "creditors", "debt restructuring", "restructuring support agreement",
    "missed payment", "missed coupon", "skipped payment", "downgrade to junk", "cut to junk", "junk",
    "credit rating", "rating downgrade", "downgraded by moody's", "moody's", "fitch", "rating agency",
    "credit watch", "creditwatch", "outlook negative", "liquidity crisis", "bank run", "bailout",
    "receivership", "fdic", "distressed debt", "credit spreads", "spreads widen", "covenant breach",
    "debt crisis", "solvency", "write-down", "writedown", "non-performing", "bad loans", "loan losses",
    "margin call", "collapse of", "liquidation", "went into administration",
]
OPERATIONAL = [
    "outage", "outages", "cyberattack", "cyber attack", "cyber-attack", "hack", "hacked", "hackers",
    "ransomware", "data breach", "breach", "recall", "recalls", "recalled", "strike", "strikes",
    "walkout", "explosion", "fire at", "plant fire", "crash", "grounded", "grounding", "faulty",
    "defect", "defects", "safety probe", "production halt", "halts production", "shutdown",
    "supply chain disruption", "shortage", "shortages", "hurricane", "earthquake", "flood",
    "wildfire", "spill", "accident", "malfunction", "glitch", "it failure", "system failure",
]
PRODUCT = [
    "launch", "launches", "launched", "unveil", "unveils", "unveiled", "introduces", "introduced",
    "debut", "debuts", "rolls out", "rollout", "roll out", "new product", "releases", "released",
    "fda approval", "fda approves", "approved by fda", "gets approval", "wins approval",
    "announces new", "next-generation", "next generation", "new model", "new chip", "new iphone",
    "preorder", "pre-order", "goes on sale", "available today",
]
LEGAL = [
    "lawsuit", "lawsuits", "sued", "sues", "court", "judge", "ruling", "verdict", "settlement",
    "antitrust", "probe", "investigation", "investigates", "regulator", "regulators", "sec charges",
    "doj", "department of justice", "ftc", "fined", "penalty", "class action", "subpoena",
    "indicted", "fraud", "compliance", "regulation", "regulatory", "ban", "banned", "european commission",
]
MNA = [
    "acquire", "acquires", "acquired", "acquisition", "merger", "merge", "merges", "takeover",
    "buyout", "to buy", "deal to buy", "bid for", "stake in", "divest", "divests", "spin off",
    "spinoff", "spin-off", "tender offer", "hostile bid", "agreed to purchase",
]
LEADERSHIP = [
    "ceo", "chief executive", "cfo", "chief financial officer", "steps down", "resigns", "resigned",
    "resignation", "appointed", "appoints", "names new", "successor", "fired", "ousted", "retire",
    "retires", "board chair", "chairman",
]
EARNINGS = [
    "earnings", "quarterly results", "q1", "q2", "q3", "q4", "revenue", "profit", "eps", "guidance",
    "forecast", "beats estimates", "misses estimates", "outlook", "net income", "quarterly",
]
MACRO = [
    "inflation", "cpi", "ppi", "gdp", "recession", "unemployment", "jobless", "payrolls",
    "interest rate", "interest rates", "rate hike", "rate cut", "federal reserve", "the fed", "fomc",
    "powell", "ecb", "bank of england", "bank of japan", "central bank", "treasury yields",
    "bond yields", "yield curve", "dollar", "currency", "oil prices", "crude", "opec", "commodities",
    "stagflation", "consumer spending", "retail sales", "manufacturing pmi", "pmi", "housing market",
    "economy", "economic", "deficit", "debt ceiling", "stimulus", "quantitative",
]

GEOPOLITICAL_RX = _rx(GEOPOLITICAL)
CREDIT_RX = _rx(CREDIT)
OPERATIONAL_RX = _rx(OPERATIONAL)
PRODUCT_RX = _rx(PRODUCT)
LEGAL_RX = _rx(LEGAL)
MNA_RX = _rx(MNA)
LEADERSHIP_RX = _rx(LEADERSHIP)
EARNINGS_RX = _rx(EARNINGS)
MACRO_RX = _rx(MACRO)

# ---------------------------------------------------------------- high-precision rule overrides
# (event, pattern) - when a pattern fires the event is forced regardless of the classifier.
# Kept deliberately narrow: each one is close to unambiguous in financial text.
OVERRIDES: list[tuple[str, re.Pattern]] = [
    ("Credit Event", re.compile(r"\b(files?|filed|filing)\s+for\s+(chapter\s+(11|7)|bankruptcy)\b|"
                                r"\bchapter\s+11\b|\b(missed|misses|skipped)\s+(a\s+)?(debt|bond|coupon|interest)\s+payment|"
                                r"\b(downgrade[sd]?|cut[s]?)\b.{0,40}\bto\s+junk\b|\bdefault(ed|s)?\s+on\s+(its\s+)?(debt|bonds?|loans?)\b|"
                                r"\bbank\s+run\b|\bplaced\s+into\s+receivership\b", re.I)),
    ("Merger & Acquisition", re.compile(r"\b(agree[sd]?|deal|offer[s]?|bid[s]?)\s+to\s+(acquire|buy|purchase)\b|"
                                        r"\b(acquisition|takeover|merger)\s+of\b|\ball-(cash|stock)\s+deal\b", re.I)),
    ("Leadership Change", re.compile(r"\b(ceo|chief executive|cfo|chairman)\b.{0,40}\b(steps?\s+down|resign(s|ed)?|"
                                     r"retir(es|ing)|ousted|fired|to\s+depart)\b|\bnames?\s+new\s+(ceo|chief executive|cfo)\b", re.I)),
    ("Operational Risk", re.compile(r"\b(ransomware|cyber-?\s?attack|data\s+breach|global\s+outage|recall[s]?\s+\d[\d,]*|"
                                    r"grounds?\s+(all|its)?\s*(fleet|planes|jets))\b", re.I)),
    ("Geopolitical", re.compile(r"\b(invad(e|es|ed)|invasion\s+of|declares?\s+war|missile\s+strikes?|imposes?\s+sanctions|"
                                r"new\s+sanctions\s+on|tariffs?\s+on\s+(all\s+)?(imports|chinese|china|eu|mexico|canada))\b", re.I)),
    ("Macroeconomic", re.compile(r"\b(fed|federal reserve|ecb|bank of england|central bank|fomc)\b.{0,40}"
                                 r"\b(raises?|hikes?|cuts?|holds?|lowers?)\b.{0,25}\b(rates?|interest)\b", re.I)),
]

# ---------------------------------------------------------------- impact severity cues
# weight in [0, 1]; the impact model max-pools the weights of the cues present.
SEVERITY_CUES: dict[str, float] = {
    "bankruptcy": 1.0, "chapter 11": 1.0, "default": 0.95, "defaulted": 0.95, "collapse": 0.9,
    "collapses": 0.9, "invasion": 1.0, "invades": 1.0, "war": 0.85, "declares war": 1.0,
    "nuclear": 0.9, "missile": 0.8, "sanctions": 0.7, "embargo": 0.75, "tariffs": 0.65,
    "trade war": 0.75, "crisis": 0.8, "crash": 0.8, "plunge": 0.75, "plunges": 0.75, "plummets": 0.75,
    "tumbles": 0.6, "soars": 0.55, "surges": 0.55, "skyrockets": 0.6, "record high": 0.45,
    "record low": 0.6, "emergency": 0.85, "emergency rate cut": 1.0, "rate hike": 0.6,
    "rate cut": 0.55, "recession": 0.75, "stagflation": 0.75, "downgrade": 0.55, "downgraded": 0.55,
    "junk": 0.8, "fraud": 0.8, "indicted": 0.75, "probe": 0.45, "investigation": 0.45,
    "antitrust": 0.5, "lawsuit": 0.4, "fined": 0.45, "billion": 0.35, "trillion": 0.5,
    "recall": 0.55, "outage": 0.55, "cyberattack": 0.7, "ransomware": 0.7, "data breach": 0.6,
    "grounded": 0.7, "layoffs": 0.45, "job cuts": 0.45, "resigns": 0.5, "steps down": 0.5,
    "ousted": 0.6, "profit warning": 0.7, "cuts guidance": 0.65, "slashes": 0.6, "halts": 0.55,
    "suspends": 0.55, "bank run": 1.0, "bailout": 0.85, "contagion": 0.85, "liquidity": 0.5,
    "selloff": 0.6, "sell-off": 0.6, "turmoil": 0.7, "shock": 0.7, "escalation": 0.7,
    "escalates": 0.7, "ceasefire": 0.5, "acquisition": 0.45, "merger": 0.45, "takeover": 0.5,
    "beats estimates": 0.35, "misses estimates": 0.45, "all-time high": 0.45, "volatility": 0.4,
}
SEVERITY_RX = _rx(list(SEVERITY_CUES))
PCT_RX = re.compile(r"(\d{1,3}(?:\.\d+)?)\s?(%|percent|per cent)", re.I)
MONEY_RX = re.compile(r"\$\s?(\d+(?:\.\d+)?)\s?(billion|bn|b|trillion|tn|t)\b", re.I)

# ---------------------------------------------------------------- regions (for Module B targeting)
REGIONS: dict[str, list[str]] = {
    "North America": ["united states", "u.s.", "us economy", "america", "american", "canada", "canadian",
                       "mexico", "mexican", "washington", "wall street"],
    "Europe": ["europe", "european", "eurozone", "euro zone", "eu", "germany", "german", "france", "french",
               "italy", "italian", "spain", "uk", "britain", "british", "england", "london", "brussels"],
    "China": ["china", "chinese", "beijing", "shanghai", "hong kong", "yuan", "renminbi", "xi jinping"],
    "Asia-Pacific": ["japan", "japanese", "tokyo", "korea", "korean", "taiwan", "taiwanese", "india", "indian",
                     "australia", "singapore", "indonesia", "vietnam", "asia", "asian"],
    "Middle East": ["middle east", "israel", "israeli", "iran", "iranian", "saudi", "gaza", "yemen", "houthi",
                    "red sea", "strait of hormuz", "opec", "qatar", "uae", "iraq", "syria", "lebanon"],
    "Russia/CIS": ["russia", "russian", "moscow", "kremlin", "putin", "ukraine", "ukrainian", "kyiv", "belarus"],
    "Latin America": ["brazil", "brazilian", "argentina", "chile", "colombia", "venezuela", "peru", "latin america"],
}
REGION_RX = {region: _rx(terms) for region, terms in REGIONS.items()}

# ---------------------------------------------------------------- sector / market-wide linking
SECTORS: dict[str, list[str]] = {
    "Financials": ["banks", "banking sector", "bank stocks", "lenders", "financial stocks", "insurers"],
    "Energy": ["oil stocks", "energy stocks", "oil majors", "oil producers", "drillers", "refiners", "crude oil",
               "oil prices", "opec", "natural gas"],
    "Information Technology": ["tech stocks", "chipmakers", "semiconductor", "semiconductors", "chip stocks",
                               "big tech", "software stocks", "ai stocks"],
    "Health Care": ["drugmakers", "pharma stocks", "pharmaceutical", "biotech", "healthcare stocks", "vaccine makers"],
    "Industrials": ["airlines", "aerospace", "defense stocks", "industrial stocks", "manufacturers"],
    "Consumer Staples": ["consumer staples", "retailers", "grocers", "beverage makers"],
    "Consumer Discretionary": ["automakers", "carmakers", "ev makers", "e-commerce", "consumer discretionary"],
    "Communication Services": ["social media companies", "media stocks", "streaming", "advertisers"],
}
SECTOR_RX = {sector: _rx(terms) for sector, terms in SECTORS.items()}

MARKET_WIDE = [
    "stock market", "stocks", "equities", "wall street", "s&p 500", "s&p500", "dow jones", "the dow",
    "nasdaq", "russell 2000", "global markets", "markets", "investors", "bond market", "treasuries",
    "risk assets", "vix", "volatility index",
] + MACRO
MARKET_RX = _rx(MARKET_WIDE)

# ---------------------------------------------------------------- source credibility
TIER1_NEWS = {"reuters", "bloomberg", "wsj", "wall street journal", "financial times", "ft.com", "cnbc",
              "associated press", "apnews", "ap news", "barrons", "marketwatch", "economist", "nytimes",
              "new york times", "washington post", "bbc", "fortune", "forbes", "yahoo finance", "nikkei"}


@lru_cache(maxsize=4096)
def credibility(source_type: str, publisher: str, author: str = "") -> float:
    """1.0 tier-1 newswire, 0.8 other news, 0.65 curated finance-news accounts, 0.5 open social."""
    name = (publisher or "").lower()
    if source_type == "news":
        return 1.0 if any(t in name for t in TIER1_NEWS) else 0.8
    if name == "x-corpus":  # the tweet corpus is drawn from finance-news accounts
        return 0.65
    return 0.5


def hits(rx: re.Pattern, text: str) -> list[str]:
    return [m.group(0).lower() for m in rx.finditer(text)]
