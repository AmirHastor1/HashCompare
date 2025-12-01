
import matplotlib.pyplot as plt
import numpy as np

n_list = []
build_list = []
query_list = []

with open('hash_benchmarks.csv', 'r') as f:
    lines = f.readlines()[1:]
    for line in lines:
        parts = line.strip().split(',')
        n_list.append(float(parts[0]))
        build_list.append(float(parts[1]))
        query_list.append(float(parts[2]))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

ax1.plot(n_list, build_list, 'o-', linewidth=2, markersize=8, label='Universal Chain')
ax1.set_xlabel('Broj kljuceva (n)', fontsize=12)
ax1.set_ylabel('Vrijeme gradjenja (s)', fontsize=12)
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_title('Universal Hashing - Vrijeme kreiranja', fontsize=14)
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(n_list, query_list, 's-', linewidth=2, markersize=8, label='Universal Chain')
ax2.set_xlabel('Broj kljuceva (n)', fontsize=12)
ax2.set_ylabel('Vrijeme pretrage (s)', fontsize=12)
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_title('Universal Hashing - Vrijeme pretrage', fontsize=14)
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('hash_benchmark_plots.png', dpi=300, bbox_inches='tight')
plt.show()

print("Grafovi snimljeni u: hash_benchmark_plots.png")
