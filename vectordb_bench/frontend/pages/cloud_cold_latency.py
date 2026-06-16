from pathlib import Path
from html import escape

import streamlit as st

from vectordb_bench import config
from vectordb_bench.frontend.components.check_results.footer import footer
from vectordb_bench.frontend.components.check_results.headerIcon import drawHeaderIcon
from vectordb_bench.frontend.components.check_results.nav import NavToPages
from vectordb_bench.frontend.components.cloudleaderboard.cold_latency import (
    CloudColdLatencyRow,
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

CASE_TITLE = "Cloud Cold Latency Case"
CASE_DESCRIPTION = (
    "This case measures the first query after an idle cold period against the warmed steady-state query path. "
    "It isolates cold-start behavior from normal search throughput so the chart shows whether a product has "
    "a material warm-up penalty after inactivity."
)

PRODUCT_COLORS = {
    "pinecone_serverless": "#e64f78",
    "turbopuffer": "#f3702d",
    "turbopuffer_pinned": "#9a52ca",
    "zilliz_cloud_cap_12cu": "#5bc7d5",
    "zilliz_cloud_capacity_12cu": "#5bc7d5",
    "zilliz_cloud_tiered_4cu": "#43a657",
}

PRODUCT_WARM_COLORS = {
    "pinecone_serverless": "#f6a1b9",
    "turbopuffer": "#ffc27e",
    "turbopuffer_pinned": "#d9a4ed",
    "zilliz_cloud_cap_12cu": "#a8ecf1",
    "zilliz_cloud_capacity_12cu": "#a8ecf1",
    "zilliz_cloud_tiered_4cu": "#9bd8a5",
}


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

    rows = load_cloud_cold_latency_rows(CLOUD_COLD_LATENCY_RAW_RESULTS_DIR)
    if not rows:
        st.info(
            "No Cloud Cold Latency result files are available yet. "
            f"Expected raw files under `{CLOUD_COLD_LATENCY_RAW_RESULTS_DIR}`."
        )
        footer(st.container())
        return

    st.markdown(render_cold_latency_intro_html(), unsafe_allow_html=True)
    select_column, _ = st.columns([1, 4])
    with select_column:
        selected_mode = st.selectbox(
            "Mode",
            mode_options(rows),
            format_func=lambda mode: mode,
            key="cloud-cold-latency-mode",
        )

    records = cold_latency_case_view_records(rows, selected_mode)
    if not records:
        st.warning("No rows match the selected mode.")
        footer(st.container())
        return

    st.markdown(
        render_cold_latency_case_html(records, selected_mode, include_intro=False),
        unsafe_allow_html=True,
    )

    footer(st.container())


def mode_options(rows: list[CloudColdLatencyRow]) -> list[str]:
    return sorted({row.mode_key for row in rows}, key=_mode_sort_key)


def cold_latency_case_view_records(rows: list[CloudColdLatencyRow], mode_key: str) -> list[dict]:
    selected_rows = [row for row in rows if row.mode_key == mode_key]
    if not selected_rows:
        return []
    max_latency_ms = max(
        (_seconds_to_ms(row.cold_first_query_seconds) + _seconds_to_ms(row.warm_first_query_seconds))
        for row in selected_rows
    )
    max_ratio = max((row.first_query_ratio or 0) for row in selected_rows)
    records = []
    for row in selected_rows:
        cold_ms = _seconds_to_ms(row.cold_first_query_seconds)
        warm_ms = _seconds_to_ms(row.warm_first_query_seconds)
        first_query_ratio = round(row.first_query_ratio or 0, 2)
        records.append(
            {
                "product_key": row.product_key,
                "product_name": _case_product_name(row),
                "mode_key": row.mode_key,
                "mode_display": row.mode_display,
                "status": row.status,
                "cold_ms": cold_ms,
                "warm_ms": warm_ms,
                "first_query_ratio": first_query_ratio,
                "latency_cold_width": _bar_width(cold_ms, max_latency_ms),
                "latency_warm_width": _bar_width(warm_ms, max_latency_ms),
                "ratio_width": _bar_width(first_query_ratio, max_ratio),
                "color": PRODUCT_COLORS.get(row.product_key, "#64748b"),
                "warm_color": PRODUCT_WARM_COLORS.get(row.product_key, "#cbd5e1"),
            }
        )
    return sorted(records, key=lambda record: record["cold_ms"])


def render_cold_latency_intro_html() -> str:
    return f"""
    <style>{_CLOUD_COLD_LATENCY_CSS}</style>
    <div class="cloud-cold-intro">
      <h1>{escape(CASE_TITLE)}</h1>
      <p>{escape(CASE_DESCRIPTION)}</p>
    </div>
    """


def render_cold_latency_case_html(
    records: list[dict],
    selected_mode: str,
    *,
    include_intro: bool = True,
) -> str:
    intro = render_cold_latency_intro_html() if include_intro else f"<style>{_CLOUD_COLD_LATENCY_CSS}</style>"
    latency_rows = "\n".join(_render_latency_row(record) for record in records)
    ratio_rows = "\n".join(_render_ratio_row(record) for record in records)
    return f"""
    {intro}
    <div class="cloud-cold-mode-readout">
      <div class="cloud-cold-mode-label">Mode</div>
      <div class="cloud-cold-mode-value">{escape(selected_mode)}</div>
    </div>
    <div class="cloud-cold-card-grid">
      <section class="cloud-cold-card">
        <div class="cloud-cold-card-header">
          <h2>Cold / Warm Latency</h2>
          <span>{escape(selected_mode)}</span>
        </div>
        <div class="cloud-cold-rows">{latency_rows}</div>
      </section>
      <section class="cloud-cold-card">
        <div class="cloud-cold-card-header">
          <h2>Cold / Warm Ratio</h2>
          <span>lower is better</span>
        </div>
        <div class="cloud-cold-rows">{ratio_rows}</div>
      </section>
    </div>
    <section class="cloud-cold-notes">
      <h3>Notes:</h3>
      <ol>
        <li>We note that while certain products may have a more dramatic cold/warm ratio at p99 percentile, this usually indicates a network shaking problem in later queries and cannot be fully reproduced. Thus we stick with the more faithful definition of cold/warm latency, i.e. the first query for each round.</li>
        <li>The timing for when a product's collection becomes cold is rather ambiguous since most products don't offer public APIs to provide such info. In order to simulate real world production settings, for cold latency benchmarking, we ensure to wait at least 24 hours since the last operations on the products for the collections to become as cold as possible.</li>
      </ol>
    </section>
    """


def _render_latency_row(record: dict) -> str:
    product = escape(record["product_name"])
    status_title = f' title="{escape(record["status"])}"'
    return f"""
    <div class="cloud-cold-row"{status_title}>
      <div class="cloud-cold-product">{product}</div>
      <div class="cloud-cold-bar-track">
        <span class="cloud-cold-bar cold" style="width:{record["latency_cold_width"]}%; background:{record["color"]};"></span>
        <span class="cloud-cold-bar warm" style="width:{record["latency_warm_width"]}%; background:{record["warm_color"]};"></span>
      </div>
      <div class="cloud-cold-value"><strong>{record["cold_ms"]} / {record["warm_ms"]}</strong><span>ms</span></div>
    </div>
    """


def _render_ratio_row(record: dict) -> str:
    product = escape(record["product_name"])
    return f"""
    <div class="cloud-cold-row">
      <div class="cloud-cold-product">{product}</div>
      <div class="cloud-cold-bar-track">
        <span class="cloud-cold-ratio-bar" style="width:{record["ratio_width"]}%; background:{record["color"]};"></span>
      </div>
      <div class="cloud-cold-value"><strong>{record["first_query_ratio"]:.2f}x</strong></div>
    </div>
    """


def _mode_sort_key(value: str) -> tuple[int, str]:
    if value in {"Unfiltered", "unfiltered"}:
        return (0, value)
    return (1, value)


def _seconds_to_ms(seconds: float) -> int:
    return int(seconds * 1000 + 0.5)


def _bar_width(value: float, max_value: float) -> float:
    if max_value <= 0:
        return 0
    return round(max(1, min(100, value / max_value * 100)), 2)


def _case_product_name(row: CloudColdLatencyRow) -> str:
    if row.product_key == "turbopuffer_pinned":
        return "Turbopuffer Pinned"
    return row.product_name


_CLOUD_COLD_LATENCY_CSS = """
.cloud-cold-intro {
  color: #06283d;
  margin-top: 1.25rem;
}
.cloud-cold-intro h1 {
  font-size: 2.15rem;
  line-height: 1.2;
  margin: 0 0 1rem;
  font-weight: 760;
}
.cloud-cold-intro p {
  color: #697891;
  font-size: 1.05rem;
  line-height: 1.5;
  margin: 0 0 1.75rem;
  max-width: 76rem;
}
.cloud-cold-mode-readout {
  margin: 0.5rem 0 1.75rem;
  display: none;
}
.cloud-cold-mode-label {
  color: #697891;
  font-weight: 700;
  margin-bottom: 0.45rem;
}
.cloud-cold-mode-value {
  border: 1px solid #e3e8ef;
  border-radius: 8px;
  width: 16rem;
  padding: 0.62rem 0.85rem;
  color: #111827;
  background: #ffffff;
}
.cloud-cold-card-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 1.5rem;
  margin-top: 1.35rem;
}
.cloud-cold-card {
  border: 1px solid #e3e8ef;
  border-radius: 8px;
  padding: 1.55rem 1.85rem;
  background: #ffffff;
}
.cloud-cold-card-header {
  display: flex;
  justify-content: space-between;
  gap: 1rem;
  align-items: baseline;
  margin-bottom: 1.3rem;
}
.cloud-cold-card-header h2 {
  color: #06283d;
  font-size: 1.15rem;
  line-height: 1.25;
  margin: 0;
  font-weight: 760;
}
.cloud-cold-card-header span {
  color: #697891;
  font-size: 0.98rem;
}
.cloud-cold-rows {
  display: flex;
  flex-direction: column;
  gap: 1.45rem;
}
.cloud-cold-row {
  display: grid;
  grid-template-columns: minmax(12rem, 1.3fr) minmax(9rem, 1fr) minmax(5.5rem, auto);
  gap: 1.1rem;
  align-items: center;
}
.cloud-cold-product {
  color: #06283d;
  font-size: 1rem;
  line-height: 1.2;
  font-weight: 760;
}
.cloud-cold-bar-track {
  height: 1.2rem;
  background: #f5f7fa;
  display: flex;
  overflow: hidden;
}
.cloud-cold-bar,
.cloud-cold-ratio-bar {
  display: block;
  height: 100%;
}
.cloud-cold-value {
  color: #06283d;
  display: flex;
  align-items: baseline;
  justify-content: flex-end;
  gap: 0.45rem;
  white-space: nowrap;
  font-size: 0.86rem;
}
.cloud-cold-value strong {
  font-size: 1.02rem;
}
.cloud-cold-value span {
  color: #697891;
}
.cloud-cold-notes {
  margin-top: 1.5rem;
  background: #f5f7fa;
  border-radius: 8px;
  padding: 1.45rem 1.65rem;
  color: #697891;
}
.cloud-cold-notes h3 {
  color: #06283d;
  margin: 0 0 0.75rem;
  font-size: 1.08rem;
}
.cloud-cold-notes ol {
  margin: 0;
  padding-left: 1.3rem;
}
.cloud-cold-notes li {
  margin-bottom: 0.65rem;
  line-height: 1.45;
}
@media (max-width: 960px) {
  .cloud-cold-card-grid {
    grid-template-columns: 1fr;
  }
  .cloud-cold-row {
    grid-template-columns: minmax(8rem, 1fr);
    gap: 0.55rem;
  }
  .cloud-cold-value {
    justify-content: flex-start;
  }
}
"""


if __name__ == "__main__":
    main()
