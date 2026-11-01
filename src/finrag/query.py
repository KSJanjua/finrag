"""Query understanding: which reporting periods does a question ask about?

Questions name periods in many ways - "Q3 2024", "Q3'24", "3Q-2024", "Q3 FY2024",
"third quarter of 2024", "quarter three, 2024", "Q1, Q2 and Q3 2024" - and
full-year figures ("full-year 2024", "in 2024") are reported in that year's Q4
report. The parser turns all of these into (fiscal_year, quarter) pairs so the
retriever can filter by report or split the question per period.
"""
import re
from dataclasses import dataclass, field

from . import config

ORDINALS = {"first": 1, "second": 2, "third": 3, "fourth": 4, "1st": 1, "2nd": 2, "3rd": 3, "4th": 4,
            "one": 1, "two": 2, "three": 3, "four": 4}
_YEAR = r"(?:fy\s*|fiscal\s+(?:year\s+)?)?'?((?:20)?\d{2})(?!\d)"
_ORD = r"(first|second|third|fourth|1st|2nd|3rd|4th)"

PATTERNS = [
    # Q3 2024, Q3-2024, Q3'24, Q3 FY2024, Q3 of 2024
    re.compile(r"\bq([1-4])[\s\-']*(?:of\s+)?" + _YEAR, re.I),
    # 3Q-2024, 3Q24, 3Q 2024
    re.compile(r"\b([1-4])q[\s\-']*" + _YEAR, re.I),
    # third quarter of 2024, 3rd quarter 2024
    re.compile(r"\b" + _ORD + r"\s+quarter\s*(?:of\s+|,\s*)?" + _YEAR, re.I),
    # quarter 3 2024, quarter three, 2024
    re.compile(r"\bquarter\s+([1-4]|one|two|three|four)\s*(?:of\s+|,\s*)?" + _YEAR, re.I),
]
# "Q1, Q2 and Q3 2024": quarters that share the year written after the last one.
QUARTER_LIST = re.compile(r"\b((?:q[1-4]\s*(?:,|and|&|to|-)\s*)+)q([1-4])[\s\-']*" + _YEAR, re.I)
FULL_YEAR = re.compile(r"\b(?:full[\s-]year|fiscal\s+year|annual|fy)\s*'?((?:20)\d{2})\b", re.I)
BARE_YEAR = re.compile(r"\b(?:in|for|during|of)\s+((?:20)\d{2})\b", re.I)


def _year(text):
    y = int(text)
    return y + 2000 if y < 100 else y


def _quarter(text):
    return int(text) if text.isdigit() else ORDINALS[text.lower()]


@dataclass
class QueryPeriods:
    quarters: list = field(default_factory=list)   # [(fiscal_year, quarter)] in order of mention
    full_years: list = field(default_factory=list)  # [fiscal_year] asked about as a whole year

    def periods(self):
        """All periods, with a full year mapped to its Q4 (annual figures are in the Q4 report)."""
        out = list(self.quarters)
        for y in self.full_years:
            if (y, 4) not in out:
                out.append((y, 4))
        return out


def parse_periods(query):
    """Reporting periods mentioned in a question, in order of first mention."""
    quarters, spans = [], []

    def add(period, span):
        spans.append(span)
        if period not in quarters:
            quarters.append(period)

    for m in QUARTER_LIST.finditer(query):
        year = _year(m.group(3))
        for q in re.findall(r"q([1-4])", m.group(1), re.I):
            add((year, int(q)), m.span())
        add((year, int(m.group(2))), m.span())
    for pattern in PATTERNS:
        for m in pattern.finditer(query):
            add((_year(m.group(2)), _quarter(m.group(1))), m.span())

    def outside_quarter_mentions(m):
        return not any(a <= m.start() < b for a, b in spans)

    full_years = [int(m.group(1)) for m in FULL_YEAR.finditer(query) if outside_quarter_mentions(m)]
    if not quarters and not full_years:
        full_years = [int(m.group(1)) for m in BARE_YEAR.finditer(query)]
    full_years = list(dict.fromkeys(full_years))
    return QueryPeriods(quarters=quarters, full_years=full_years)


def docs_for_periods(periods):
    """Registered report for each period (None where the corpus has no report of its own)."""
    return [config.DOC_BY_PERIOD.get(p) for p in periods]
