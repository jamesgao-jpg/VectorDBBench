"""
Cost-per-hour calculator for LAION 100M sustained search across
Turbopuffer, Zilliz Cloud Tiered, and Zilliz Cloud Capacity.

Assumes linear CU -> QPS scaling for Zilliz (measured at reference CU count).
Turbopuffer cost is based on bytes-queried per query (namespace size).

Usage:
    python cost_calc.py                # default QPS = 1, 10, 100
    python cost_calc.py 5 50 500       # custom QPS list
"""
import sys

# --- Turbopuffer pricing ---
TPUF_RETURNED_PER_GB = 0.05           # $/GB returned
TPUF_QUERIED_BASE_PER_PB = 1.00       # $/PB at full rate (0-32 GB/query)
TPUF_TIER2_DISCOUNT = 0.80            # 80% off for 32-128 GB/query marginal
TPUF_TIER3_DISCOUNT = 0.96            # 96% off for >128 GB/query marginal
TPUF_MIN_QUERIED_GB = 1.28            # floor per query

# --- Zilliz pricing ---
ZILLIZ_TIERED_PER_CU_HR = 0.372
ZILLIZ_CAPACITY_PER_CU_HR = 0.248

# --- Measured reference points on LAION 100M (k=100, no filter) ---
TIERED_REF_CU = 4
TIERED_REF_QPS = 4.29
CAPACITY_REF_CU = 12
CAPACITY_REF_QPS = 310.47

# --- CU floor for LAION 100M ---
# 4 CU Tiered and 12 CU Capacity are the smallest cluster sizes we have
# tested holding the ~300 GB LAION 100M dataset. Sub-floor counts may or
# may not hold the data — untested. For Tiered, 4 CU suffices at read
# time because cold data sits in object storage and CUs only cache hot
# working set; for Capacity, 12 CU was the smallest we measured and the
# minimum viable size for this dataset is likely lower but not verified.
TIERED_MIN_CU = 4
CAPACITY_MIN_CU = 12

# --- LAION 100M namespace size (as Turbopuffer's JS bills it) ---
# NOTE: TB's JS uses decimal GB (/1e9), not binary GiB (/1024**3).
# NOTE: queried-bytes uses dim_bytes = 2 (f16-equivalent index scan),
#       NOT physical f32 (4 B). Storage billing uses f32 separately.
DATASET_N = 100_000_000
DIM = 768
VEC_BYTES_QUERIED = 2  # TB bills queries on the quantized index (2 B/dim)
VEC_BYTES_STORED = 4   # f32 storage (for the (un-implemented) storage line)
ATTR_BYTES = 0         # LAION has no extra attributes beyond pk+vector
PK_BYTES = 8           # not counted by TB's n() helper
NAMESPACE_GB = DATASET_N * (ATTR_BYTES + DIM * VEC_BYTES_QUERIED) / 1e9

# --- Returned bytes per query (top-100 IDs + distances, ~20 B each) ---
RETURNED_BYTES_PER_QUERY = 100 * 20


def tpuf_cost_per_query(namespace_gb: float = NAMESPACE_GB,
                        returned_bytes: float = RETURNED_BYTES_PER_QUERY) -> float:
    """Turbopuffer: per-query cost based on bytes queried (namespace size)."""
    queried_gb = max(namespace_gb, TPUF_MIN_QUERIED_GB)

    # Tiered pricing on queried bytes (all in $/GB: $1/PB = $1e-6/GB)
    t1 = min(queried_gb, 32)
    t2 = min(max(queried_gb - 32, 0), 96)
    t3 = max(queried_gb - 128, 0)
    base_rate = TPUF_QUERIED_BASE_PER_PB / 1e6  # $/GB
    queried_cost = (
        t1 * base_rate
        + t2 * base_rate * (1 - TPUF_TIER2_DISCOUNT)
        + t3 * base_rate * (1 - TPUF_TIER3_DISCOUNT)
    )

    returned_cost = returned_bytes / 1e9 * TPUF_RETURNED_PER_GB
    return queried_cost + returned_cost


def tpuf_cost_per_hour(qps: float) -> float:
    return qps * 3600 * tpuf_cost_per_query()


def zilliz_cost_per_hour(qps: float, ref_cu: int, ref_qps: float,
                         per_cu_hr: float, min_cu: int = 1,
                         integer_cu: bool = True) -> tuple[float, float]:
    """Return (cost_per_hour, CU_used). `min_cu` is the storage-capacity floor."""
    qps_per_cu = ref_qps / ref_cu
    cu_needed = qps / qps_per_cu
    if integer_cu:
        cu_needed = max(min_cu, -(-cu_needed // 1))  # ceil, min min_cu
    else:
        cu_needed = max(min_cu, cu_needed)
    return cu_needed * per_cu_hr, cu_needed


def main():
    qps_list = [float(a) for a in sys.argv[1:]] or [1, 10, 100]

    print(f"LAION 100M namespace: {NAMESPACE_GB:.1f} GB")
    print(f"Turbopuffer per-query cost: ${tpuf_cost_per_query():.2e}")
    print(f"Zilliz Tiered: {TIERED_REF_QPS / TIERED_REF_CU:.2f} QPS/CU at ${ZILLIZ_TIERED_PER_CU_HR}/CU/hr")
    print(f"Zilliz Capacity: {CAPACITY_REF_QPS / CAPACITY_REF_CU:.2f} QPS/CU at ${ZILLIZ_CAPACITY_PER_CU_HR}/CU/hr")
    print()

    header = f"{'QPS':>6} | {'Turbopuffer':>14} | {'Tiered (CU)':>20} | {'Capacity (CU)':>20}"
    print(header)
    print("-" * len(header))

    for qps in qps_list:
        tp = tpuf_cost_per_hour(qps)
        tier_cost, tier_cu = zilliz_cost_per_hour(
            qps, TIERED_REF_CU, TIERED_REF_QPS, ZILLIZ_TIERED_PER_CU_HR,
            min_cu=TIERED_MIN_CU, integer_cu=True,
        )
        cap_cost, cap_cu = zilliz_cost_per_hour(
            qps, CAPACITY_REF_CU, CAPACITY_REF_QPS, ZILLIZ_CAPACITY_PER_CU_HR,
            min_cu=CAPACITY_MIN_CU, integer_cu=True,
        )
        print(f"{qps:>6.0f} | ${tp:>12.3f}/hr | ${tier_cost:>10.3f}/hr ({int(tier_cu):>3} CU) | ${cap_cost:>10.3f}/hr ({int(cap_cu):>3} CU)")


if __name__ == "__main__":
    main()
