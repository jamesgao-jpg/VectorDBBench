from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from vectordb_bench import config
from vectordb_bench.frontend.components.check_results.footer import footer
from vectordb_bench.frontend.components.check_results.headerIcon import drawHeaderIcon
from vectordb_bench.frontend.components.check_results.nav import NavToPages
from vectordb_bench.frontend.config.styles import FAVICON, PAGE_TITLE
from vectordb_bench.frontend.components.cloudleaderboard.multi_tenant_search import (
    MultiTenantSearchRow,
    load_multi_tenant_search_rows,
    multi_tenant_search_records,
)


MULTI_TENANT_SEARCH_RAW_RESULTS_DIR = (
    Path(config.RESULTS_LOCAL_DIR) / "cloudleaderboard" / "cloud_multi_tenant_search" / "raw_results"
)
PAGE_CASE_TITLE = "Cloud Multi-Tenant Search"
PAGE_CASE_CAPTION = "Multi-tenant search case"
PAGE_HEADER_TITLE = "Cloud Multi-Tenant Search"
PAGE_HEADER_CAPTION = "Hosted vector database multi-tenant search results."


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

    rows = load_multi_tenant_search_rows(MULTI_TENANT_SEARCH_RAW_RESULTS_DIR)
    if not rows:
        st.info(
            "No Cloud Multi-Tenant Search result files are available yet. "
            f"Expected raw files under `{MULTI_TENANT_SEARCH_RAW_RESULTS_DIR}`."
        )
        footer(st.container())
        return

    shown_rows = _filter_rows(rows)
    records = multi_tenant_search_records(shown_rows)
    if not records:
        st.warning("No rows match the selected filters.")
        footer(st.container())
        return

    df = pd.DataFrame.from_records(records)
    st.dataframe(df, hide_index=True, use_container_width=True)

    _draw_charts_by_search_mode(st, records)
    _draw_detail_section(st, shown_rows)

    footer(st.container())


def _filter_rows(rows: list[MultiTenantSearchRow]) -> list[MultiTenantSearchRow]:
    st.sidebar.header(PAGE_CASE_TITLE)
    st.sidebar.caption(PAGE_CASE_CAPTION)

    products = sorted({row.product_name for row in rows})
    search_modes = sorted({row.search_mode for row in rows})
    filters = sorted({row.filter_display for row in rows}, key=_filter_sort_key)
    payloads = sorted({row.payload_profile for row in rows})
    concurrency_signatures = sorted({row.concurrency_signature for row in rows}, key=_concurrency_sort_key)

    selected_products = st.sidebar.multiselect("Products", products, default=products)
    selected_modes = st.sidebar.multiselect(
        "Search Modes",
        search_modes,
        default=search_modes,
        format_func=_search_mode_display,
    )
    selected_filters = st.sidebar.multiselect("Filter Rates", filters, default=filters)
    selected_payloads = st.sidebar.multiselect(
        "Payloads",
        payloads,
        default=payloads,
        format_func=_payload_display,
    )
    selected_concurrency = st.sidebar.multiselect(
        "Concurrency",
        concurrency_signatures,
        default=concurrency_signatures,
    )

    return [
        row
        for row in rows
        if row.product_name in selected_products
        and row.search_mode in selected_modes
        and row.filter_display in selected_filters
        and row.payload_profile in selected_payloads
        and row.concurrency_signature in selected_concurrency
    ]


def _draw_qps_chart(container, df: pd.DataFrame):
    filter_label, payload_label = _chart_selection_labels(df)
    fig = px.bar(
        df,
        x="Max QPS",
        y="Product",
        color="Payload",
        hover_data=["Filter", "Tenant Count", "Top K", "Concurrency", "Best Concurrency"],
        orientation="h",
        title=f"Max QPS - {filter_label} / {payload_label}",
    )
    fig.update_layout(margin=dict(l=0, r=0, t=48, b=0), legend_title_text="")
    container.plotly_chart(fig, width="stretch")


def _draw_latency_chart(container, df: pd.DataFrame):
    filter_label, payload_label = _chart_selection_labels(df)
    latency_df = df.melt(
        id_vars=["Product", "Search Mode", "Filter", "Payload", "Concurrency"],
        value_vars=["P95 Latency (s)", "P99 Latency (s)"],
        var_name="Latency Metric",
        value_name="Latency (s)",
    ).dropna(subset=["Latency (s)"])
    fig = px.bar(
        latency_df,
        x="Latency (s)",
        y="Product",
        color="Latency Metric",
        hover_data=["Filter", "Payload", "Concurrency"],
        orientation="h",
        title=f"Latency at Best Concurrency - {filter_label} / {payload_label}",
    )
    fig.update_layout(margin=dict(l=0, r=0, t=48, b=0), legend_title_text="")
    container.plotly_chart(fig, width="stretch")


def _draw_detail_section(container, rows: list[MultiTenantSearchRow]):
    detail = container.expander("Per-concurrency details", expanded=False)
    detail_records = []
    for row in rows:
        for index, concurrency in enumerate(row.concurrency):
            detail_records.append(
                {
                    "Product": row.product_name,
                    "Search Mode": _search_mode_display(row.search_mode),
                    "Filter": row.filter_display,
                    "Payload": _payload_display(row.payload_profile),
                    "Tenant Count": row.tenant_count,
                    "Top K": row.top_k,
                    "Concurrency Group": row.concurrency_signature,
                    "Concurrency": concurrency,
                    "QPS": _value_at(row.concurrency_qps, index),
                    "Avg Latency (s)": _value_at(row.concurrency_latency_avg, index),
                    "P95 Latency (s)": _value_at(row.concurrency_latency_p95, index),
                    "P99 Latency (s)": _value_at(row.concurrency_latency_p99, index),
                }
            )
    if detail_records:
        detail.dataframe(pd.DataFrame.from_records(detail_records), hide_index=True, use_container_width=True)
    else:
        detail.caption("No per-concurrency details are available for the selected rows.")


def _draw_charts_by_search_mode(container, records: list[dict]):
    grouped = chart_records_by_search_mode(records)
    tabs = container.tabs(list(grouped.keys()))
    for tab, (search_mode, mode_records) in zip(tabs, grouped.items()):
        filters = sorted({record["Filter"] for record in mode_records}, key=_filter_sort_key)
        control_columns = tab.columns(2)
        selected_filter = control_columns[1].selectbox(
            "Chart Filter Rate",
            filters,
            key=f"cloud-mt-chart-filter-{search_mode}",
        )
        filter_records = [
            record for record in mode_records if record["Filter"] == selected_filter
        ]
        payloads = sorted({record["Payload"] for record in filter_records})
        selected_payload = control_columns[0].selectbox(
            "Chart Payload",
            payloads,
            key=f"cloud-mt-chart-payload-{search_mode}",
        )
        selected_records = chart_records_for_selection(
            mode_records,
            search_mode=search_mode,
            filter_display=selected_filter,
            payload=selected_payload,
        )
        if not selected_records:
            tab.caption("No chart rows match this selection.")
            continue

        chart_df = pd.DataFrame.from_records(selected_records)
        chart_columns = tab.columns(2)
        _draw_qps_chart(chart_columns[0], chart_df)
        _draw_latency_chart(chart_columns[1], chart_df)


def chart_records_by_search_mode(records: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for record in records:
        grouped.setdefault(record["Search Mode"], []).append(record)
    return grouped


def chart_records_for_selection(
    records: list[dict],
    *,
    search_mode: str,
    filter_display: str,
    payload: str,
) -> list[dict]:
    return [
        record
        for record in records
        if record["Search Mode"] == search_mode
        and record["Filter"] == filter_display
        and record["Payload"] == payload
    ]


def _filter_sort_key(value: str):
    if value == "Unfiltered":
        return -1.0
    if value == "Unknown":
        return 999.0
    return float(value.rstrip("%"))


def _concurrency_sort_key(value: str):
    if value == "unknown":
        return (999999,)
    return tuple(int(part.removeprefix("c")) for part in value.split(","))


def _search_mode_display(search_mode: str) -> str:
    return {
        "unfiltered": "Unfiltered",
        "int_filter": "Integer Filter",
        "scalar_label_filter": "Scalar Label Filter",
    }.get(search_mode, search_mode)


def _payload_display(payload_profile: str) -> str:
    return {
        "ids_only": "IDs Only",
        "scalar_label": "Scalar Label",
        "vector": "Vector",
    }.get(payload_profile, payload_profile)


def _value_at(values, index: int):
    if index >= len(values):
        return None
    return values[index]


def _chart_selection_labels(df: pd.DataFrame) -> tuple[str, str]:
    filter_label = str(df["Filter"].iloc[0]) if "Filter" in df and not df.empty else "Selected Filter"
    payload_label = str(df["Payload"].iloc[0]) if "Payload" in df and not df.empty else "Selected Payload"
    return filter_label, payload_label


if __name__ == "__main__":
    main()
