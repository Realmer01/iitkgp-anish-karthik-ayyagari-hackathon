"""Framing: does a headline report an event, or discuss one?

Questions ("Can US farms withstand a trade war?"), explainers ("... Here's what it means"), listicles
("3 stocks that could benefit") and hedged speculation ("... could trigger a recession") talk about
possible events rather than report one. Module B uses this so that only reported events trigger a
stress test. The impact score deliberately does not: tested as an impact feature it lowered the
event-study correlation (explainers cluster on volatile days - "Why is the stock market down today?").
"""
from __future__ import annotations

import re

QUESTION_OPENER = re.compile(r"^\W*(?:is|are|was|were|will|would|can|could|should|does|do|did|has|have|shall)\s+"
                             r"(?!not\b)", re.I)
WH_OPENER = re.compile(r"^\W*(?:why|how|what|when|where|who|which)\b", re.I)
EXPLAINER = re.compile(
    r"\b(?:explained|explainer|what (?:it|this|that) (?:is|means)|what (?:you|investors|traders) (?:need to|should) know|"
    r"what to (?:know|expect|watch)|here'?s (?:why|what|how)|how to (?:prepare|invest|protect|profit|survive|play)|"
    r"things to (?:know|watch)|in charts|lessons from|to understand|opinion|op-ed|podcast|live updates?)\b", re.I)
LISTICLE = re.compile(r"^\W*(?:\d{1,2}|two|three|four|five|six|seven|eight|nine|ten)\s+(?:[\w'-]+\s+){0,3}?"
                      r"(?:stocks?|ways|things|reasons|charts|signs|lessons|picks|etfs|questions|takeaways)\b", re.I)
HEDGE = re.compile(r"\b(?:could|might|would)\s+(?!not\b)\w+|(?<!\bin )(?<!\bof )(?<!\bby )(?<!\bsince )(?<!\bfrom )"
                   r"(?<!\buntil )(?<!\bearly )(?<!\blate )\bmay\s+(?:be|face|hit|trigger|cause|benefit|force|lead|"
                   r"spark|send|push|derail|tip|see|soon|need|struggle|slow|rise|fall|drop|climb|come|get|take|start|"
                   r"end|remain|stay|keep|have|not|want|lose|gain|cut|raise|hike|delay|pause)\b", re.I)


def factuality(text: str) -> tuple[float, str]:
    """1.0 = reported event; 0.5 per softening cue (explanatory framing, hedge); 0.0 = question / explainer /
    listicle. Returns the score and a short label for the strongest cue ("" when reported)."""
    t = (text or "").strip()
    if t.endswith("?") or QUESTION_OPENER.search(t):
        return 0.0, "question"
    if EXPLAINER.search(t):
        return 0.0, "explainer"
    if LISTICLE.search(t):
        return 0.0, "listicle"
    score, labels = 1.0, []
    if "?" in t or WH_OPENER.search(t):
        score, labels = score * 0.5, labels + ["explanatory"]
    if HEDGE.search(t):
        score, labels = score * 0.5, labels + ["hedged"]
    return score, "+".join(labels)
