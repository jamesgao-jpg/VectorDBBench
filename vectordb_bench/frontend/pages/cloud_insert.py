from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from vectordb_bench import config
from vectordb_bench.frontend.components.check_results.footer import footer
from vectordb_bench.frontend.components.check_results.headerIcon import drawHeaderIcon
from vectordb_bench.frontend.components.check_results.nav import NavToPages
from vectordb_bench.frontend.components.cloudleaderboard.insert import (
    CloudInsertRow,
    cloud_insert_records,
    load_cloud_insert_rows,
)
from vectordb_bench.frontend.config.styles import FAVICON, PAGE_TITLE


CLOUD_INSERT_RAW_RESULTS_DIR = (
    Path(config.RESULTS_LOCAL_DIR) / "cloudleaderboard" / "cloud_insert" / "raw_results"
)
PAGE_CASE_TITLE = "Cloud Insert"
PAGE_CASE_CAPTION = "Insert readiness case"
PAGE_HEADER_TITLE = "Cloud Insert"
PAGE_HEADER_CAPTION = "Hosted vector database insert readiness results."


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

    rows = load_cloud_insert_rows(CLOUD_INSERT_RAW_RESULTS_DIR)
    if not rows:
        st.info(
            "No Cloud Insert result files are available yet. "
            f"Expected raw files under `{CLOUD_INSERT_RAW_RESULTS_DIR}`."
        )
        footer(st.container())
        return

    shown_rows = _filter_rows(rows)
    records = cloud_insert_records(shown_rows)
    if not records:
        st.warning("No rows match the selected filters.")
        footer(st.container())
        return

    df = pd.DataFrame.from_records(records)
    st.dataframe(df, hide_index=True, use_container_width=True)

    _draw_charts(st, records)

    footer(st.container())


def _filter_rows(rows: list[CloudInsertRow]) -> list[CloudInsertRow]:
    st.sidebar.header(PAGE_CASE_TITLE)
    st.sidebar.caption(PAGE_CASE_CAPTION)

    products = sorted({row.product_name for row in rows})
    modes = sorted({row.mode_display for row in rows})
    datasets = sorted({row.dataset for row in rows})
    batch_sizes = sorted({row.batch_size for row in rows})

    selected_products = st.sidebar.multiselect("Products", products, default=products)
    selected_modes = st.sidebar.multiselect("Modes", modes, default=modes)
    selected_datasets = st.sidebar.multiselect("Datasets", datasets, default=datasets)
    selected_batches = st.sidebar.multiselect("Batch Sizes", batch_sizes, default=batch_sizes)

    return [
        row
        for row in rows
        if row.product_name in selected_products
        and row.mode_display in selected_modes
        and row.dataset in selected_datasets
        and row.batch_size in selected_batches
    ]


def _draw_charts(container, records: list[dict]):
    grouped = chart_records_by_batch_size(records)
    tabs = container.tabs([str(batch_size) for batch_size in grouped])
    for tab, (batch_size, batch_records) in zip(tabs, grouped.items()):
        batch_df = pd.DataFrame.from_records(batch_records)
        chart_columns = tab.columns(2)
        _draw_throughput_chart(chart_columns[0], batch_df, batch_size)
        _draw_timing_chart(chart_columns[1], batch_df, batch_size)


def _draw_throughput_chart(container, df: pd.DataFrame, batch_size: int):
    fig = px.bar(
        df,
        x="Insert Rows/s",
        y="Product",
        color="Mode",
        hover_data=["Dataset", "Inserted Count", "Total Readiness (s)"],
        orientation="h",
        title=f"Insert Throughput - Batch {batch_size}",
    )
    fig.update_layout(margin=dict(l=0, r=0, t=48, b=0), legend_title_text="")
    container.plotly_chart(fig, width="stretch")


def _draw_timing_chart(container, df: pd.DataFrame, batch_size: int):
    timing_df = df.melt(
        id_vars=["Product", "Mode", "Dataset", "Batch Size"],
        value_vars=["Insert Completion (s)", "Searchable Delay (s)", "Indexed Delay (s)"],
        var_name="Timing Component",
        value_name="Seconds",
    )
    fig = px.bar(
        timing_df,
        x="Seconds",
        y="Product",
        color="Timing Component",
        hover_data=["Mode", "Dataset"],
        orientation="h",
        title=f"Readiness Timing - Batch {batch_size}",
    )
    fig.update_layout(margin=dict(l=0, r=0, t=48, b=0), legend_title_text="")
    container.plotly_chart(fig, width="stretch")


def chart_records_by_batch_size(records: list[dict]) -> dict[int, list[dict]]:
    grouped: dict[int, list[dict]] = {}
    for record in records:
        grouped.setdefault(record["Batch Size"], []).append(record)
    return dict(sorted(grouped.items()))


if __name__ == "__main__":
    main()
