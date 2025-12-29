
import csv
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

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

# Sort by n for each algo
for algo in data:
    zipped = list(zip(data[algo]['n'], data[algo]['build'], data[algo]['query'], data[algo]['metric']))
    zipped.sort(key=lambda x: x[0])
    data[algo]['n'], data[algo]['build'], data[algo]['query'], data[algo]['metric'] = map(list, zip(*zipped))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

for algo in sorted(data.keys()):
    ax1.plot(data[algo]['n'], data[algo]['build'], marker='o', linewidth=2, label=algo)
    ax2.plot(data[algo]['n'], data[algo]['query'], marker='s', linewidth=2, label=algo)

ax1.set_xlabel('Broj kljuceva (n)')
ax1.set_ylabel('Vrijeme gradjenja (s)')
ax1.set_title('Build time (sekunde)')

ax2.set_xlabel('Broj kljuceva (n)')
ax2.set_ylabel('Vrijeme pretrage (s)')
ax2.set_title('Query time (sekunde)')

# x log je OK (n varira puno), ali y ostaje LINEAR da ne dobijas 10^x na vremenu
ax1.set_xscale('log')
ax2.set_xscale('log')

# Force plain seconds formatting (bez scientific 1eX / 10^X)
for ax in (ax1, ax2):
    ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    ax.ticklabel_format(style='plain', axis='y', useOffset=False)
    ax.grid(True, alpha=0.3)
    ax.legend()

plt.tight_layout()
plt.savefig('hash_benchmark_plots.png', dpi=300, bbox_inches='tight')
plt.show()

print("Grafovi snimljeni u: hash_benchmark_plots.png")
