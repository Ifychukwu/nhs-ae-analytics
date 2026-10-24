"""A&E Data Explorer: a Streamlit app over NHS England's Hospital A&E Activity data.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from ae_data import assistant, views
from ae_data.glossary import DIMENSION_NOTES, MEASURE_NOTES
from ae_data.query import QueryError, connect, run_select

SOURCE_URL = "https://digital.nhs.uk/data-and-information/publications/statistical/hospital-accident--emergency-activity"

MEASURE_LABELS = {
    "attendances": "Attendances",
    "rate_per_100k": "Rate per 100,000 population",
    "pct_of_total": "% of all attendances",
    "population": "Population",
}

st.set_page_config(page_title="A&E Data Explorer", page_icon="🏥", layout="wide")


# --------------------------------------------------------------------------
# Data access (cached)
# --------------------------------------------------------------------------

@st.cache_resource
def get_con():
    return connect()


@st.cache_data
def cached(name: str, *args):
    """Cache any view function by name and arguments."""
    return getattr(views, name)(get_con(), *args)


def fmt_int(v) -> str:
    return "–" if pd.isna(v) else f"{v:,.0f}"


def fmt_pct(v) -> str:
    return "–" if pd.isna(v) else f"{v:.1f}%"


# --------------------------------------------------------------------------
# Layout
# --------------------------------------------------------------------------

st.title("A&E Data Explorer")
st.caption(f"England's emergency department activity, April 2020 to March 2026. "
           f"Source: NHS England, [Hospital Accident & Emergency Activity]({SOURCE_URL}) (ECDS).")

tab_overview, tab_explore, tab_trusts, tab_ask, tab_about = st.tabs(
    ["Overview", "Explore", "Trusts", "Ask the data", "About the data"])


# ---- Overview -------------------------------------------------------------
with tab_overview:
    head = cached("headline")
    latest, prev = head.iloc[-1], head.iloc[-2]
    st.subheader(f"{latest.financial_year} at a glance")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Attendances", fmt_int(latest.attendances),
              f"{100 * (latest.attendances / prev.attendances - 1):+.1f}% vs {prev.financial_year}")
    c2.metric("Within 4 hours", fmt_pct(latest.pct_within_4h),
              f"{latest.pct_within_4h - prev.pct_within_4h:+.1f} pts")
    c3.metric("Over 12 hours in A&E", fmt_pct(latest.pct_over_12h),
              f"{latest.pct_over_12h - prev.pct_over_12h:+.1f} pts", delta_color="inverse")
    c4.metric("Admitted", fmt_pct(latest.pct_admitted), f"{latest.pct_admitted - prev.pct_admitted:+.1f} pts",
              delta_color="off")

    m = cached("monthly")
    fig = px.line(m, x="date", y="attendances", markers=True,
                  labels={"date": "", "attendances": "Attendances per month"})
    fig.update_traces(hovertemplate="%{x|%b %Y}<br>%{y:,.0f} attendances<extra></extra>")
    fig.update_layout(title="Monthly A&E attendances", height=380, margin=dict(t=50, b=10))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("The low point in April 2020 reflects the first lockdown.")

    left, right = st.columns(2)
    with left:
        f2 = go.Figure()
        f2.add_bar(x=head.financial_year, y=head.pct_within_4h, name="Within 4 hours (%)",
                   hovertemplate="%{x}<br>%{y:.1f}% within 4 hours<extra></extra>")
        f2.update_layout(title="Seen within 4 hours", yaxis=dict(range=[0, 100], title="%"), height=340,
                         margin=dict(t=50, b=10))
        st.plotly_chart(f2, use_container_width=True)
    with right:
        f3 = px.line(head, x="financial_year", y="pct_over_12h", markers=True,
                     labels={"financial_year": "", "pct_over_12h": "% over 12 hours"})
        f3.update_layout(title="Spent over 12 hours in A&E", height=340, margin=dict(t=50, b=10))
        st.plotly_chart(f3, use_container_width=True)


# ---- Explore -------------------------------------------------------------
with tab_explore:
    st.subheader("Break attendances down by any recorded characteristic")
    dims = cached("dimensions")
    c1, c2, c3 = st.columns([2, 2, 1])
    dim = c1.selectbox("Characteristic", dims, index=dims.index("Chief complaint") if "Chief complaint" in dims else 0,
                       key="ex_dim")
    measures = cached("measures_for", dim)
    measure = c2.selectbox("Measure", measures, format_func=lambda m: MEASURE_LABELS.get(m, m), key="ex_measure")
    yrs = cached("years")
    year = c3.selectbox("Year", yrs, index=len(yrs) - 1, key="ex_year")
    st.caption(DIMENSION_NOTES.get(dim, ""))

    c4, c5 = st.columns([3, 1])
    search = c4.text_input("Filter categories (optional)", key="ex_search",
                           placeholder="e.g. chest, fracture, ambulance")
    top = c5.slider("Show top", 5, 50, 15, key="ex_top")

    data = cached("breakdown", dim, measure, year, top, search)
    if data.empty:
        st.info("No rows match. Try another year or clear the filter.")
    else:
        fig = px.bar(data.iloc[::-1], x="value", y="category", orientation="h",
                     labels={"value": MEASURE_LABELS.get(measure, measure), "category": ""})
        fig.update_layout(height=max(320, 26 * len(data) + 80), margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.download_button("Download these rows (CSV)", data.to_csv(index=False).encode(),
                           file_name=f"ae_{dim}_{measure}_{year}.csv".replace(" ", "_").lower(), mime="text/csv")

        st.markdown("**Change over time**")
        by = "month" if cached("has_months", dim) and st.toggle("Show by month", key="ex_monthly") else "year"
        picked = st.multiselect("Categories", data["category"].tolist(), default=data["category"].tolist()[:5],
                                key="ex_cats")
        tr = cached("trend", dim, measure, picked, by)
        if not tr.empty:
            ft = px.line(tr, x="period", y="value", color="category", markers=True,
                         labels={"period": "", "value": MEASURE_LABELS.get(measure, measure), "category": ""})
            ft.update_layout(height=400, legend=dict(orientation="h", y=-0.25), margin=dict(t=10))
            st.plotly_chart(ft, use_container_width=True)


# ---- Trusts --------------------------------------------------------------
with tab_trusts:
    st.subheader("How a trust compares with England")
    tr = cached("trusts")
    names = tr["org_name"].tolist()
    default = names.index("Manchester University NHS Foundation Trust") if \
        "Manchester University NHS Foundation Trust" in names else 0
    name = st.selectbox("Trust", names, index=default, key="tr_name")
    code = tr.loc[tr["org_name"] == name, "org_code"].iloc[0]

    w = cached("trust_waits", code)
    cols = st.columns(len(w)) if len(w) else []
    for col, r in zip(cols, w.itertuples()):
        col.markdown(f"**{r.financial_year}** · {fmt_int(r.attendances)} attendances")
        a, b = col.columns(2)
        a.metric("Within 4 hours", fmt_pct(r.pct_within_4h),
                 None if pd.isna(r.england_pct_within_4h) else f"{r.pct_within_4h - r.england_pct_within_4h:+.1f} pts vs England")
        b.metric("Over 12 hours", fmt_pct(r.pct_over_12h),
                 None if pd.isna(r.england_pct_over_12h) else f"{r.pct_over_12h - r.england_pct_over_12h:+.1f} pts vs England",
                 delta_color="inverse")
    st.caption("England here is the sum of all trusts in the same file, so trust and benchmark share one basis. "
               "In 2023-24 trust waiting times cover 94% of attendances, so compare each year with its own "
               "England figure rather than across years.")

    pyears = cached("provider_years")
    c1, c2 = st.columns([2, 1])
    mix_dim = c1.selectbox("Compare case mix by", views.MIX_DIMENSIONS,
                           index=views.MIX_DIMENSIONS.index("Acuity"), key="tr_mix")
    mix_year = c2.selectbox("Year", pyears, index=len(pyears) - 1, key="tr_year")
    mix = cached("trust_mix", code, mix_dim, mix_year)
    if mix.empty:
        st.info("No figures for this trust and characteristic.")
    else:
        long = mix.melt(id_vars="category", value_vars=["trust_pct", "england_pct"],
                        var_name="who", value_name="pct")
        long["who"] = long["who"].map({"trust_pct": name, "england_pct": "England"})
        fm = px.bar(long, x="category", y="pct", color="who", barmode="group",
                    labels={"pct": "% of attendances", "category": "", "who": ""})
        fm.update_layout(height=380, legend=dict(orientation="h", y=1.1), margin=dict(t=30))
        st.plotly_chart(fm, use_container_width=True)
        st.caption("Counts of 1 to 7 are suppressed by NHS England, so small categories may be missing.")

    st.markdown(f"**All trusts with a major A&E, {mix_year}** (sorted by share over 12 hours)")
    lg = cached("league", mix_year)
    st.dataframe(
        lg.rename(columns={"org_name": "Trust", "attendances": "Attendances", "pct_within_4h": "Within 4h %",
                           "pct_over_12h": "Over 12h %", "pct_admitted": "Admitted %"}).drop(columns="org_code"),
        use_container_width=True, hide_index=True,
        column_config={"Attendances": st.column_config.NumberColumn(format="%d"),
                       "Within 4h %": st.column_config.NumberColumn(format="%.1f"),
                       "Over 12h %": st.column_config.NumberColumn(format="%.1f"),
                       "Admitted %": st.column_config.NumberColumn(format="%.1f")})


# ---- Ask the data ---------------------------------------------------------
def api_key() -> str | None:
    try:
        key = st.secrets.get("ANTHROPIC_API_KEY")
    except Exception:
        key = None
    return key or os.environ.get("ANTHROPIC_API_KEY")


def show_runs(runs):
    for i, run in enumerate(runs, 1):
        with st.expander(f"Query {i}: {run.purpose or 'data lookup'}"):
            st.code(run.sql, language="sql")
            if run.error:
                st.error(run.error)
            elif run.result is not None:
                st.dataframe(run.result, use_container_width=True, hide_index=True)


with tab_ask:
    st.subheader("Ask a question in plain English")
    key = api_key()
    if key:
        import anthropic

        client = anthropic.Anthropic(api_key=key)
        st.caption("Answers are written by Claude from SQL queries on this data. "
                   "Every query is shown so you can check the numbers.")
        if "chat" not in st.session_state:
            st.session_state.chat = []
        for turn in st.session_state.chat:
            with st.chat_message(turn["role"]):
                st.markdown(turn["content"])
                if turn.get("runs"):
                    show_runs(turn["runs"])
        if not st.session_state.chat:
            st.markdown("Try: *Which age group has the highest A&E attendance rate?* · "
                        "*How did Blackpool compare with England in 2024-25?* · "
                        "*Which chief complaints grew most since 2022-23?*")
        q = st.chat_input("Ask about attendances, waiting times, trusts, age, deprivation…")
        if q:
            with st.chat_message("user"):
                st.markdown(q)
            history = [{"role": t["role"], "content": t["content"]} for t in st.session_state.chat][-8:]
            with st.chat_message("assistant"):
                with st.spinner("Querying the data…"):
                    try:
                        ans = assistant.ask(q, client, get_con(), history=history)
                    except Exception as e:  # network or API errors
                        ans = assistant.Answer(text=f"Sorry, the question could not be answered: {e}")
                st.markdown(ans.text)
                show_runs(ans.queries)
            st.session_state.chat += [{"role": "user", "content": q},
                                      {"role": "assistant", "content": ans.text, "runs": ans.queries}]
        if st.session_state.chat and st.button("Clear conversation"):
            st.session_state.chat = []
            st.rerun()
    else:
        st.info("Plain-English questions need an Anthropic API key: set ANTHROPIC_API_KEY as an environment "
                "variable or in `.streamlit/secrets.toml`. Meanwhile, pick a ready-made question or write SQL.")
        choice = st.selectbox("Ready-made questions", list(assistant.EXAMPLES), key="ask_example")
        sql = st.text_area("SQL (read-only)", assistant.EXAMPLES[choice], height=150, key=f"sql_{choice}")
        if st.button("Run query", type="primary"):
            try:
                res = run_select(sql, get_con())
                st.dataframe(res, use_container_width=True, hide_index=True)
                if res.attrs.get("truncated"):
                    st.caption("Showing the first 5,000 rows.")
                st.download_button("Download results (CSV)", res.to_csv(index=False).encode(),
                                   file_name="ae_query.csv", mime="text/csv")
            except QueryError as e:
                st.error(str(e))


# ---- About ------------------------------------------------------------------
with tab_about:
    st.subheader("About the data")
    st.markdown(f"""
**Source.** NHS England, [Hospital Accident & Emergency Activity]({SOURCE_URL}): the Emergency Care Data Set
(ECDS) open data files for 2022-23 to 2025-26 (England, monthly from April 2020) and the provider level
analysis files for 2023-24 and 2024-25. Published under the Open Government Licence.

**How the files were combined.** Each yearly file repeats earlier years, sometimes revised and relabelled.
For every characteristic and period the app keeps the newest publication's figures. Labels that differ only
in spelling or case are merged. Diagnoses are kept by year only, to keep the database small.

**Checks run before the database is built.** No duplicate rows; monthly attendances add up to each year's
published total; published 12 hour counts and percentages agree; percentages lie between 0 and 100; trust
totals add up to England's.

**Things to know.**
- ECDS figures are published as experimental and come from provider submissions.
- The 4 hour percentage excludes sites piloting new standards; the 12 hour percentage uses all attendances.
- Trust waiting times for 2023-24 cover 94% of England's attendances (99.7% in 2024-25).
- Counts of 1 to 7 are suppressed in trust files and are not stored.
- Ethnicity is 'Not stated' or 'Unknown' for a large share of attendances.
""")
    g1, g2 = st.columns(2)
    g1.markdown("**Characteristics**")
    g1.dataframe(pd.DataFrame(DIMENSION_NOTES.items(), columns=["Characteristic", "What it is"]),
                 hide_index=True, use_container_width=True)
    g2.markdown("**Measures**")
    g2.dataframe(pd.DataFrame(MEASURE_NOTES.items(), columns=["Measure", "What it is"]),
                 hide_index=True, use_container_width=True)
