
import csv
import matplotlib.pyplot as plt

CSV_FILE = "hash_bench_all.csv"

rows = []
with open(CSV_FILE, "r", newline="") as f:
    r = csv.DictReader(f)
    for row in r:
        row["n"] = int(row["n"])
        row["fill_time_s"] = float(row["fill_time_s"])
        row["query_time_s"] = float(row["query_time_s"])
        rows.append(row)

dtypes = sorted(set(r["dtype"] for r in rows))
algos = sorted(set(r["algo"] for r in rows))

def series(dtype, algo):
    pts = [r for r in rows if r["dtype"] == dtype and r["algo"] == algo]
    pts.sort(key=lambda x: x["n"])
    n = [p["n"] for p in pts]
    fill = [p["fill_time_s"] for p in pts]
    query = [p["query_time_s"] for p in pts]
    return n, fill, query

def plot_one_dtype(dtype: str):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), squeeze=False)

    handles = []
    labels = []

    for algo in algos:
        n, fill, query = series(dtype, algo)

        l, = axes[0][0].plot(n, fill, marker="o", linewidth=2, label=algo)
        axes[0][1].plot(n, query, marker="o", linewidth=2, label=algo)

        axes[1][0].plot(n, fill, marker="o", linewidth=2, label=algo)
        axes[1][1].plot(n, query, marker="o", linewidth=2, label=algo)

        handles.append(l)
        labels.append(algo)

    axes[0][0].set_title("Popunjavanje (linear)")
    axes[0][1].set_title("Query (linear)")
    for ax in (axes[0][0], axes[0][1]):
        ax.set_xscale("log")
        ax.grid(True, which="both", alpha=0.3)
        ax.set_xlabel("Broj kljuceva (n)")
        ax.set_ylabel("Vrijeme (s)")

    axes[1][0].set_title("Popunjavanje (log-n)")
    axes[1][1].set_title("Query (log-n)")
    for ax in (axes[1][0], axes[1][1]):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.3)
        ax.set_xlabel("Broj kljuceva (n)")
        ax.set_ylabel("Vrijeme (s)")

    fig.suptitle(f"Hash benchmark za tip: {dtype}", y=0.98)
    fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 4))

    fig.tight_layout(rect=[0, 0.08, 1, 0.95])
    fig.subplots_adjust(hspace=0.35, wspace=0.25)

    out_png = f"hash_bench_{dtype}.png"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.show()
    print("Saved:", out_png)

for dtype in dtypes:
    plot_one_dtype(dtype)
