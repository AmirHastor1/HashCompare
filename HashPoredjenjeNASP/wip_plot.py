
import matplotlib.pyplot as plt
data = {}
with open('wip_hash.csv') as f:
    for line in f.readlines()[1:]:
        n, b, q = line.split(',')
        plt.plot(float(n), float(b), 'o-', label='build')
        plt.plot(float(n), float(q), 's-', label='query')
plt.xscale('log'); plt.yscale('log')
plt.savefig('wip_plot.png')
