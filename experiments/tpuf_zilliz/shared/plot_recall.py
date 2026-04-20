"""Plot recall vs wall-clock time for the Cohere 10M streaming bp=OFF test."""
import matplotlib.pyplot as plt

# (time_seconds, recall, label)
data = [
    (92,   0.0000, "10%"),
    (174,  0.0000, "20%"),
    (253,  0.0631, "30%"),
    (335,  0.1236, "40%"),
    (416,  0.1779, "50%"),
    (497,  0.1779, "60%"),
    (579,  0.1779, "70%"),
    (661,  0.2344, "80%"),
    (743,  0.2907, "90%"),
    (827,  0.3309, "100%"),
    (884,  0.3439, "110% (hint_warm)"),
    (1654, 0.8363, "fully indexed"),
]

times = [d[0] for d in data]
recalls = [d[1] for d in data]
labels = [d[2] for d in data]

fig, ax = plt.subplots(figsize=(11, 6))
ax.plot(times, recalls, marker="o", linewidth=2, markersize=7, color="#1f77b4")

for t, r, lbl in data:
    ax.annotate(lbl, (t, r),
                textcoords="offset points", xytext=(6, 8),
                fontsize=9, color="#333")

# Mark the insert-ended region
ax.axvspan(0, 827, alpha=0.08, color="orange", label="insert in progress (bp=OFF)")
ax.axvspan(827, 1654, alpha=0.08, color="green", label="post-insert / indexing drain")

ax.axhline(0.8363, color="green", linestyle="--", alpha=0.5, label="ANN ceiling (0.8363)")

ax.set_xlabel("Wall-clock time (seconds from insert start)")
ax.set_ylabel("Recall@100")
ax.set_title("Cohere 10M streaming, bp=OFF + eventual consistency\nRecall vs time")
ax.grid(True, alpha=0.3)
ax.set_ylim(-0.02, 1.0)
ax.legend(loc="center right")

fig.tight_layout()
out = "/home/ubuntu/recall_vs_time.png"
fig.savefig(out, dpi=130)
print(f"saved to {out}")
