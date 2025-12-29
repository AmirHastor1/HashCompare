
import csv
import matplotlib.pyplot as plt

# CSV format: algo,n,build_time_s,query_time_s,metric
data = {}

with open('hash_benchmarks.csv', 'r', newline='') as f:
    r = csv.DictReader(f)
    for row in r:
        algo = row['algo']
        n = int(row['n'])
        build = float(row['build_time_s'])
        query = float(row['query_time_s'])
        metric = float(row['metric'])
        data.setdefault(algo, {'n': [], 'build': [], 'query': [], 'metric': []})
        data[algo]['n'].append(n)
        data[algo]['build'].append(build)
        data[algo]['query'].append(query)
        data[algo]['metric'].append(metric)

# Sort by n
for algo in data:
    zipped = list(zip(data[algo]['n'], data[algo]['build'], data[algo]['query'], data[algo]['metric']))
    zipped.sort(key=lambda x: x[0])
    data[algo]['n'], data[algo]['build'], data[algo]['query'], data[algo]['metric'] = map(list, zip(*zipped))

# --- LOGn plotovi (x i y log) ---
fig2, (bx1, bx2) = plt.subplots(1, 2, figsize=(15, 6))

for algo in sorted(data.keys()):
    bx1.plot(data[algo]['n'], data[algo]['build'], marker='o', linewidth=2, label=algo)
    bx2.plot(data[algo]['n'], data[algo]['query'], marker='s', linewidth=2, label=algo)

bx1.set_xlabel('Broj kljuceva (n)')
bx1.set_ylabel('Vrijeme gradjenja (s)')
bx1.set_title('Build time (log n)')
bx1.set_xscale('log')
bx1.set_yscale('log')

bx2.set_xlabel('Broj kljuceva (n)')
bx2.set_ylabel('Vrijeme pretrage (s)')
bx2.set_title('Query time (log n)')
bx2.set_xscale('log')
bx2.set_yscale('log')

for ax in (bx1, bx2):
    ax.grid(True, which='both', alpha=0.3)
    ax.legend()

plt.tight_layout()
plt.savefig('hash_benchmark_plots_loglog.png', dpi=300, bbox_inches='tight')
plt.show()

print("Log-log grafovi snimljeni u: hash_benchmark_plots_loglog.png")
