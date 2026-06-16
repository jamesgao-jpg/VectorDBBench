from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from vectordb_bench import config
from vectordb_bench.frontend.components.check_results.footer import footer
from vectordb_bench.frontend.components.check_results.headerIcon import drawHeaderIcon
from vectordb_bench.frontend.components.check_results.nav import NavToPages
from vectordb_bench.frontend.components.cloudleaderboard.cold_latency import (
    CloudColdLatencyRow,
    cloud_cold_latency_records,
    load_cloud_cold_latency_rows,
)
from vectordb_bench.frontend.config.styles import FAVICON, PAGE_TITLE


CLOUD_COLD_LATENCY_RAW_RESULTS_DIR = (
    Path(config.RESULTS_LOCAL_DIR) / "cloudleaderboard" / "cloud_cold_latency" / "raw_results"
)
PAGE_CASE_TITLE = "Cloud Cold Latency"
PAGE_CASE_CAPTION = "Cold/warm latency case"
PAGE_HEADER_TITLE = "Cloud Cold Latency"
PAGE_HEADER_CAPTION = "Hosted vector database cold/warm query latency results."


def main():
    st.set_page_config(
        page_title=f"{PAGE_TITLE} - {PAGE_HEADER_TITLE}",
        page_icon=FAVICON,
        layout="wide",
        initial_sidebar_state="expanded",
    )
    drawHeaderIcon(st)
    NavToPages(st)
    st.markdown(
        "<style> div[data-testid='stSidebarNav'] {display: none;} </style>",
        unsafe_allow_html=True,
    )

    st.title(PAGE_HEADER_TITLE)
    st.caption(PAGE_HEADER_CAPTION)

    rows = load_cloud_cold_latency_rows(CLOUD_COLD_LATENCY_RAW_RESULTS_DIR)
    if not rows:
        st.info(
            "No Cloud Cold Latency result files are available yet. "
            f"Expected raw files under `{CLOUD_COLD_LATENCY_RAW_RESULTS_DIR}`."
        )
        footer(st.container())
        return

    shown_rows = _filter_rows(rows)
    records = cloud_cold_latency_records(shown_rows)
    if not records:
        st.warning("No rows match the selected filters.")
        footer(st.container())
        return

    df = pd.DataFrame.from_records(records)
    st.dataframe(df, hide_index=True, use_container_width=True)

    _draw_charts_by_mode(st, records)

    footer(st.container())


def _filter_rows(rows: list[CloudColdLatencyRow]) -> list[CloudColdLatencyRow]:
    st.sidebar.header(PAGE_CASE_TITLE)
    st.sidebar.caption(PAGE_CASE_CAPTION)

    products = sorted({row.product_name for row in rows})
    modes = sorted({row.mode_display for row in rows}, key=_mode_sort_key)
    statuses = sorted({row.status for row in rows})
    payloads = sorted({row.payload_profile for row in rows})

    selected_products = st.sidebar.multiselect("Products", products, default=products)
    selected_modes = st.sidebar.multiselect("Modes", modes, default=modes)
    selected_statuses = st.sidebar.multiselect("Status", statuses, default=statuses)
    selected_payloads = st.sidebar.multiselect(
        "Payloads",
        payloads,
        default=payloads,
        format_func=_payload_display,
    )

    return [
        row
        for row in rows
        if row.product_name in selected_products
        and row.mode_display in selected_modes
        and row.status in selected_statuses
        and row.payload_profile in selected_payloads
    ]


def _draw_charts_by_mode(container, records: list[dict]):
    grouped = chart_records_by_mode(records)
    tabs = container.tabs(list(grouped.keys()))
    for tab, (mode, mode_records) in zip(tabs, grouped.items()):
        payloads = sorted({record["Payload"] for record in mode_records})
        selected_payload = tab.selectbox(
            "Chart Payload",
            payloads,
            key=f"cloud-cold-latency-chart-payload-{mode}",
        )
        selected_records = chart_records_for_selection(
            mode_records,
            mode=mode,
            payload=selected_payload,
        )
        if not selected_records:
            tab.caption("No chart rows match this payload selection.")
            continue

        chart_df = pd.DataFrame.from_records(selected_records)
        chart_columns = tab.columns(2)
        _draw_p99_chart(chart_columns[0], chart_df, mode, selected_payload)
        _draw_first_query_chart(chart_columns[1], chart_df, mode, selected_payload)
        _draw_ratio_chart(tab, chart_df, mode, selected_payload)


def _draw_p99_chart(container, df: pd.DataFrame, mode: str, payload: str):
    latency_df = df.melt(
        id_vars=["Product", "Mode", "Status", "Payload"],
        value_vars=["Cold P99 (s)", "Warm P99 (s)"],
        var_name="Latency Metric",
        value_name="Latency (s)",
    ).dropna(subset=["Latency (s)"])
    fig = px.bar(
        latency_df,
        x="Latency (s)",
        y="Product",
        color="Latency Metric",
        hover_data=["Mode", "Status", "Payload"],
        orientation="h",
        title=f"P99 Cold vs Warm - {mode} / {payload}",
    )
    fig.update_layout(margin=dict(l=0, r=0, t=48, b=0), legend_title_text="")
    container.plotly_chart(fig, width="stretch")


def _draw_first_query_chart(container, df: pd.DataFrame, mode: str, payload: str):
    first_query_df = df.sort_values("First Cold Query (s)", ascending=True)
    fig = px.bar(
        first_query_df,
        x="First Cold Query (s)",
        y="Product",
        color="Status",
        hover_data=["Mode", "Payload", "Warm First Query (s)", "First Query Ratio"],
        orientation="h",
        title=f"First Cold Query - {mode} / {payload}",
    )
    fig.update_layout(margin=dict(l=0, r=0, t=48, b=0), legend_title_text="")
    container.plotly_chart(fig, width="stretch")


def _draw_ratio_chart(container, df: pd.DataFrame, mode: str, payload: str):
    ratio_df = df.melt(
        id_vars=["Product", "Mode", "Status", "Payload"],
        value_vars=["First Query Ratio", "P99 Ratio", "Avg Ratio"],
        var_name="Ratio Metric",
        value_name="Cold/Warm Ratio",
    ).dropna(subset=["Cold/Warm Ratio"])
    fig = px.bar(
        ratio_df,
        x="Cold/Warm Ratio",
        y="Product",
        color="Ratio Metric",
        hover_data=["Mode", "Status", "Payload"],
        orientation="h",
        title=f"Cold/Warm Ratio - {mode} / {payload}",
    )
    fig.update_layout(margin=dict(l=0, r=0, t=48, b=0), legend_title_text="")
    container.plotly_chart(fig, width="stretch")


def chart_records_by_mode(records: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for record in records:
        grouped.setdefault(record["Mode"], []).append(record)
    return dict(sorted(grouped.items(), key=lambda item: _mode_sort_key(item[0])))


def chart_records_for_selection(
    records: list[dict],
    *,
    mode: str,
    payload: str,
) -> list[dict]:
    return [
        record
        for record in records
        if record["Mode"] == mode
        and record["Payload"] == payload
    ]


def _mode_sort_key(value: str) -> tuple[int, str]:
    if value == "Unfiltered":
        return (0, value)
    return (1, value)


def _payload_display(payload_profile: str) -> str:
    return {
        "ids_only": "IDs Only",
        "scalar_label": "Scalar Label",
        "vector": "Vector",
    }.get(payload_profile, payload_profile)


if __name__ == "__main__":
    main()
