"""Tests that run without Streamlit or an API key.

    python tests/test_app.py        (or: pytest tests/)
"""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ae_data import assistant, views  # noqa: E402
from ae_data.query import QueryError, connect, run_select, validate  # noqa: E402

CON = connect()


# --------------------------------------------------------------------------
# Query safety
# --------------------------------------------------------------------------

def test_only_select_allowed():
    for bad in ["DROP TABLE national", "DELETE FROM national", "SELECT 1; DROP TABLE national",
                "INSERT INTO glossary VALUES (1,2,3)", "PRAGMA writable_schema=1", "ATTACH 'x.db' AS x", ""]:
        try:
            validate(bad)
        except QueryError:
            continue
        raise AssertionError(f"accepted: {bad!r}")


def test_select_with_comments_and_cte_allowed():
    df = run_select("-- note\nWITH t AS (SELECT 1 AS a) SELECT a FROM t;", CON)
    assert df["a"].tolist() == [1]


def test_connection_is_read_only():
    # Even if validation were bypassed, the read-only connection refuses writes.
    try:
        CON.execute("CREATE TABLE x (a)")
    except Exception:
        return
    raise AssertionError("write succeeded on read-only connection")


def test_row_cap():
    df = run_select("SELECT * FROM national", CON, max_rows=10)
    assert len(df) == 10 and df.attrs["truncated"]


# --------------------------------------------------------------------------
# Data and views
# --------------------------------------------------------------------------

def test_headline_matches_published_totals():
    h = views.headline(CON).set_index("financial_year")
    assert len(h) == 6
    assert round(h.loc["2025-26", "pct_within_4h"], 2) == 72.53
    assert int(h.loc["2024-25", "attendances"]) == 25_822_265


def test_age_rates_match_counts_and_bands_merge():
    df = run_select("""SELECT a.category, a.value AS att, p.value AS pop, r.value AS rate FROM national a
        JOIN national p ON p.dimension = a.dimension AND p.period = a.period AND p.period_type = a.period_type
             AND p.category = a.category AND p.measure = 'population'
        JOIN national r ON r.dimension = a.dimension AND r.period = a.period AND r.period_type = a.period_type
             AND r.category = a.category AND r.measure = 'rate_per_100k'
        WHERE a.dimension = 'Age' AND a.measure = 'attendances'""", CON)
    assert len(df) > 0
    assert ((df["rate"] / (1e5 * df["att"] / df["pop"]) - 1).abs() <= 0.01).all()
    cats = set(run_select("SELECT DISTINCT category FROM national WHERE dimension = 'Age'", CON)["category"])
    assert "05-09" not in cats and "5-9" in cats


def test_monthly_series_complete():
    m = views.monthly(CON)
    assert len(m) == 72 and m["month"].iloc[0] == "2020-04" and m["month"].iloc[-1] == "2026-03"
    assert m["attendances"].notna().all()


def test_england_benchmark_matches_published_2024_25():
    w = run_select("SELECT pct_within_4h FROM provider_waits WHERE org_code='ENG' AND financial_year='2024-25'", CON)
    assert abs(w["pct_within_4h"].iloc[0] - 70.92) < 0.3


def test_explore_views_return_rows():
    for dim in views.dimensions(CON):
        for measure in views.measures_for(CON, dim):
            df = views.breakdown(CON, dim, measure, "2025-26", 5)
            assert "category" in df.columns
    assert len(views.breakdown(CON, "Chief complaint", "attendances", "2025-26", 5, "chest")) >= 1
    top = views.breakdown(CON, "Chief complaint", "attendances", "2025-26", 3)
    assert not top["category"].str.lower().str.startswith(("not submitted", "invalid")).any()
    with_bad = views.breakdown(CON, "Chief complaint", "attendances", "2025-26", 3, include_unrecorded=True)
    assert with_bad["category"].str.lower().str.startswith("not submitted").any()
    assert 10 < views.unrecorded_share(CON, "Chief complaint", "2025-26") < 50


def test_trust_views():
    t = views.trusts(CON)
    assert "R0A" in set(t["org_code"])
    assert len(views.trust_waits(CON, "R0A")) == 2
    mix = views.trust_mix(CON, "R0A", "Acuity", "2024-25")
    assert abs(mix["trust_pct"].sum() - 100) < 0.1
    assert len(views.league(CON, "2024-25")) > 100


def test_examples_run():
    for q, sql in assistant.EXAMPLES.items():
        assert len(run_select(sql, CON)) > 0, q


# --------------------------------------------------------------------------
# Assistant loop with a fake Claude client
# --------------------------------------------------------------------------

def _block(**kw):
    return types.SimpleNamespace(**kw)


class FakeClient:
    """Asks for one SQL query, then answers from the tool result it receives."""

    def __init__(self, sql):
        self.sql, self.calls = sql, []
        self.messages = self

    def create(self, **kw):
        self.calls.append(kw)
        last = kw["messages"][-1]["content"]
        if isinstance(last, str):
            return _block(content=[_block(type="tool_use", id="t1", input={"query": self.sql, "purpose": "test"})])
        result = last[0]["content"]
        return _block(content=[_block(type="text", text=f"Result: {result}")])


def test_assistant_runs_sql_and_answers():
    client = FakeClient("SELECT financial_year, attendances FROM annual_waits WHERE financial_year='2025-26'")
    ans = assistant.ask("How many attendances in 2025-26?", client, CON, system="test")
    assert len(ans.queries) == 1 and ans.queries[0].error is None
    assert "26639584" in ans.text
    assert client.calls[0]["tools"][0]["name"] == "run_sql"


def test_assistant_reports_bad_sql_back_to_model():
    client = FakeClient("DELETE FROM national")
    ans = assistant.ask("Delete everything", client, CON, system="test")
    assert ans.queries[0].error and "Only SELECT" in ans.text


def test_system_prompt_lists_real_labels():
    s = assistant.system_prompt(CON)
    assert "IMD decile" in s and "provider_waits" in s and "Most deprived 10%" in s


# --------------------------------------------------------------------------
# Run app.py top to bottom with stand-ins for Streamlit and Plotly
# --------------------------------------------------------------------------

def _fake_streamlit(chat_question=None):
    st = mock.MagicMock(name="streamlit")
    st.cache_data = lambda f: f
    st.cache_resource = lambda f: f
    st.secrets = {}

    def selectbox(label, options, index=0, **kw):
        options = list(options)
        return options[index] if options else None

    st.selectbox.side_effect = selectbox
    st.multiselect.side_effect = lambda label, options, default=None, **kw: list(default or [])
    st.text_input.side_effect = lambda *a, **kw: ""
    st.text_area.side_effect = lambda label, value="", **kw: value
    st.slider.side_effect = lambda label, lo, hi, value, **kw: value
    st.toggle.side_effect = lambda *a, **kw: False
    st.button.side_effect = lambda label, *a, **kw: label == "Run query"   # exercise the query path
    st.chat_input.side_effect = lambda *a, **kw: chat_question
    st.session_state = types.SimpleNamespace()
    st.session_state.__contains__ = lambda self, k: hasattr(self, k)

    def columns(spec, **kw):
        n = spec if isinstance(spec, int) else len(spec)
        cols = [mock.MagicMock() for _ in range(n)]
        for c in cols:
            c.columns.side_effect = columns
            c.selectbox.side_effect = selectbox
            c.text_input.side_effect = lambda *a, **kw: ""
            c.slider.side_effect = lambda label, lo, hi, value, **kw: value
        return cols

    st.columns.side_effect = columns
    st.tabs.side_effect = lambda names: [mock.MagicMock() for _ in names]
    st.column_config = mock.MagicMock()
    return st


class _State(types.SimpleNamespace):
    def __contains__(self, k):
        return hasattr(self, k)


def _run_app(env_key=None, chat_question=None):
    st = _fake_streamlit(chat_question)
    st.session_state = _State()
    modules = {"streamlit": st, "plotly": mock.MagicMock(), "plotly.express": mock.MagicMock(),
               "plotly.graph_objects": mock.MagicMock(), "anthropic": mock.MagicMock()}
    env = {"ANTHROPIC_API_KEY": env_key} if env_key else {}
    with mock.patch.dict(sys.modules, modules), mock.patch.dict(os.environ, env, clear=False):
        if not env_key:
            os.environ.pop("ANTHROPIC_API_KEY", None)
        src = (ROOT / "app.py").read_text()
        exec(compile(src, "app.py", "exec"), {"__name__": "__main__"})
    return st


def test_app_runs_without_key():
    st = _run_app()
    assert st.error.call_count == 0, st.error.call_args_list
    assert st.dataframe.call_count >= 1


def test_app_runs_with_key_and_question():
    fake = FakeClient("SELECT COUNT(*) AS n FROM organisations")
    with mock.patch.object(assistant, "ask", wraps=lambda q, client, con, history=None: assistant.ask(
            q, fake, con, history=history, system="test")):
        st = _run_app(env_key="test-key", chat_question="How many organisations?")
    assert st.session_state.chat[-1]["role"] == "assistant"
    assert "n" in st.session_state.chat[-1]["content"]


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
