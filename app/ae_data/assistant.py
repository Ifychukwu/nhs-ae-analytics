"""Answer plain-English questions about the A&E data with Claude.

Claude never sees the whole database. It gets the schema and the list of
real labels, writes SQL, and calls the ``run_sql`` tool; this module runs the
query read-only and returns the rows. The final answer is written only from
those rows, and every query is shown to the user so the answer can be checked.

Needs an Anthropic API key (ANTHROPIC_API_KEY). The client is passed in, so
tests can use a fake one.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import pandas as pd

from .query import SCHEMA, QueryError, describe_values, run_select

DEFAULT_MODEL = os.environ.get("AE_MODEL", "claude-sonnet-5-5")
MAX_TOOL_CALLS = 8
ROWS_SHOWN_TO_MODEL = 200

SYSTEM_TEMPLATE = """You answer questions about NHS England A&E (emergency department) data.
The data is NHS England's Hospital A&E Activity publication (ECDS), England-level from April 2020
to March 2026 and trust-level for 2023-24 and 2024-25.

{schema}

{values}

How to work:
- Always get numbers by calling run_sql. Never state a figure you did not get from a query result.
- Use the exact dimension and measure names above. Match category text with LIKE and lower().
- Prefer yearly rows (period_type = 'year') unless the question is about months or trends within a year.
- Percentages in the data are already 0-100.
- For trust questions use provider_waits for 4 and 12 hour measures, and organisations to find codes
  (search org_name with LIKE). England's trust-level benchmark is org_code 'ENG' in provider_waits.
- If the data cannot answer the question (for example a year or measure that is not there), say so
  plainly and say what it can answer instead.

How to answer:
- Lead with the answer in one or two sentences, with the numbers and the period.
- Then add a short note on how it was worked out or any caveat (for example 'Not stated' ethnicity,
  or experimental ECDS data).
- Trust waiting times for 2023-24 cover only 94% of England's attendances, so England built from
  trusts (ENG, 66.4% within 4 hours) is lower than the published national figure (69.0%). Compare a
  trust with ENG for the same year, never with annual_waits, and mention this when 2023-24 is involved.
- The 4 hour percentage excludes sites piloting new standards; the 12 hour percentage uses all attendances.
- Use UK English. Keep it under 150 words unless the user asks for more.
"""

TOOLS = [{
    "name": "run_sql",
    "description": "Run one read-only SQLite SELECT query against the A&E database and get the rows back "
                   "as JSON. Only SELECT or WITH queries are allowed. Results are capped at 200 rows; "
                   "aggregate in SQL rather than fetching raw rows.",
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "A single SQLite SELECT statement."},
            "purpose": {"type": "string", "description": "One line on what this query finds."},
        },
        "required": ["query"],
    },
}]


@dataclass
class QueryRun:
    sql: str
    purpose: str
    result: pd.DataFrame | None = None
    error: str | None = None


@dataclass
class Answer:
    text: str
    queries: list[QueryRun] = field(default_factory=list)


def system_prompt(con) -> str:
    return SYSTEM_TEMPLATE.format(schema=SCHEMA.strip(), values=describe_values(con))


def _tool_result(run: QueryRun) -> str:
    if run.error:
        return json.dumps({"error": run.error})
    df = run.result
    payload = {
        "columns": list(df.columns),
        "rows": json.loads(df.head(ROWS_SHOWN_TO_MODEL).to_json(orient="values")),
        "row_count": len(df),
    }
    if len(df) > ROWS_SHOWN_TO_MODEL or df.attrs.get("truncated"):
        payload["note"] = f"Only the first {ROWS_SHOWN_TO_MODEL} rows are shown; aggregate in SQL."
    return json.dumps(payload, default=str)


def ask(question: str, client, con, history: list[dict] | None = None,
        model: str = DEFAULT_MODEL, system: str | None = None) -> Answer:
    """Run one question through Claude, executing its SQL tool calls.

    ``history`` holds earlier turns as plain {'role', 'content'} text messages,
    so follow-up questions ("and for 2024-25?") keep their context.
    """
    system = system or system_prompt(con)
    messages = list(history or []) + [{"role": "user", "content": question}]
    runs: list[QueryRun] = []

    for _ in range(MAX_TOOL_CALLS + 1):
        resp = client.messages.create(model=model, max_tokens=1500, system=system,
                                      tools=TOOLS, messages=messages)
        tool_uses = [b for b in resp.content if getattr(b, "type", None) == "tool_use"]
        if not tool_uses or len(runs) >= MAX_TOOL_CALLS:
            text = "".join(getattr(b, "text", "") for b in resp.content if getattr(b, "type", None) == "text")
            if tool_uses:
                text = (text + "\n\n" if text else "") + "I stopped after too many queries; try a narrower question."
            return Answer(text=text.strip(), queries=runs)

        messages.append({"role": "assistant", "content": resp.content})
        results = []
        for tu in tool_uses:
            sql = tu.input.get("query", "")
            run = QueryRun(sql=sql, purpose=tu.input.get("purpose", ""))
            try:
                run.result = run_select(sql, con)
            except QueryError as e:
                run.error = str(e)
            runs.append(run)
            results.append({"type": "tool_result", "tool_use_id": tu.id, "content": _tool_result(run),
                            "is_error": run.error is not None})
        messages.append({"role": "user", "content": results})

    return Answer(text="I could not finish answering that question.", queries=runs)


# --------------------------------------------------------------------------
# Without an API key: ready-made questions with their SQL
# --------------------------------------------------------------------------

EXAMPLES = {
    "How many A&E attendances were there each year?":
        "SELECT financial_year, attendances, pct_within_4h, pct_over_12h FROM annual_waits ORDER BY financial_year",
    "Which month had the most attendances?":
        "SELECT month, attendances FROM monthly_attendances ORDER BY attendances DESC LIMIT 5",
    "What are the most common chief complaints in 2025-26?":
        "SELECT category, value AS attendances FROM national WHERE dimension = 'Chief complaint' "
        "AND measure = 'attendances' AND period_type = 'year' AND period = '2025-26' "
        "AND category NOT LIKE 'NOT SUBMITTED%' AND category NOT LIKE 'INVALID%' AND category <> 'Total' "
        "ORDER BY value DESC LIMIT 10",
    "How does the attendance rate change with deprivation?":
        "SELECT period AS financial_year, category AS imd_group, value AS rate_per_100k FROM national "
        "WHERE dimension = 'IMD decile' AND measure = 'rate_per_100k' AND period_type = 'year' "
        "AND category LIKE '%10%' ORDER BY period, category",
    "Which trusts had the highest share of 12+ hour waits in 2024-25?":
        "SELECT org_name, attendances, ROUND(pct_over_12h, 1) AS pct_over_12h, ROUND(pct_within_4h, 1) AS pct_within_4h "
        "FROM provider_waits WHERE geography_level = 'Provider' AND financial_year = '2024-25' "
        "AND attendances > 50000 ORDER BY pct_over_12h DESC LIMIT 10",
    "How did Manchester University NHS FT compare with England?":
        "SELECT financial_year, org_name, ROUND(pct_within_4h, 1) AS pct_within_4h, ROUND(pct_over_12h, 1) AS pct_over_12h "
        "FROM provider_waits WHERE org_code IN ('R0A', 'ENG') ORDER BY financial_year, org_code",
    "What share of attendances arrive by ambulance each year?":
        "SELECT a.period AS financial_year, ROUND(100.0 * a.value / t.value, 1) AS pct_by_ambulance "
        "FROM national a JOIN national t ON t.dimension = a.dimension AND t.period = a.period "
        "AND t.period_type = a.period_type AND t.measure = a.measure AND t.category = 'Total' "
        "WHERE a.dimension = 'Arrival mode' AND a.measure = 'attendances' AND a.period_type = 'year' "
        "AND a.category LIKE 'Brought in by ambulance%' ORDER BY a.period",
}
