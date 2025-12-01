#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <random>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <cstdint>
#include <cstdlib>
using namespace std;

using ll = long long;

inline uint64_t mul_mod(uint64_t a, uint64_t b, uint64_t mod) {
    uint64_t res = 0;
    a %= mod;
    while (b) {
        if (b & 1) res = (res + a) % mod;
        a = (a * 2) % mod;
        b /= 2;
    }
    return res;
}

inline uint64_t hash_mod(uint64_t a, uint64_t x, uint64_t b, uint64_t p) {
    return (mul_mod(a, x, p) + (b % p)) % p;
}

struct RNG {
    mt19937_64 gen;
    RNG(uint64_t seed = chrono::high_resolution_clock::now().time_since_epoch().count()) {
        gen.seed(seed);
    }
    uint64_t next_u64() { return gen(); }
    uint64_t next_u64_range(uint64_t lo, uint64_t hi) {
        uniform_int_distribution<uint64_t> dist(lo, hi);
        return dist(gen);
    }
};

double now_sec() {
    return chrono::duration<double>(chrono::high_resolution_clock::now().time_since_epoch()).count();
}

struct UniversalHash {
    uint64_t a, b, p, m;
    UniversalHash(uint64_t m_, uint64_t p_ = 1000000007ULL, RNG* rng = nullptr)
        : m(m_), p(p_) {
        RNG local;
        if (!rng) rng = &local;
        a = rng->next_u64_range(1, p - 1);
        b = rng->next_u64_range(0, p - 1);
    }
    inline uint64_t operator()(uint64_t x) const {
        uint64_t modp = hash_mod(a, x, b, p);
        return modp % m;
    }
};

struct UniversalHashTable {
    size_t m;
    UniversalHash h;
    vector<vector<uint64_t>> buckets;

    UniversalHashTable(size_t m_, RNG* rng = nullptr)
        : m(m_), h(m_, 1000000007ULL, rng), buckets(m_) {
    }

    void insert(uint64_t key) {
        uint64_t idx = h(key);
        auto& v = buckets[idx];
        for (auto x : v) if (x == key) return;
        v.push_back(key);
    }

    bool find(uint64_t key) const {
        uint64_t idx = h(key);
        const auto& v = buckets[idx];
        for (auto x : v) if (x == key) return true;
        return false;
    }

    double avg_chain_length() const {
        size_t cnt = 0, used = 0;
        for (auto& v : buckets) {
            if (!v.empty()) {
                used++;
                cnt += v.size();
            }
        }
        return used ? (double)cnt / used : 0.0;
    }
};

vector<uint64_t> generate_keys(size_t n, RNG& rng, bool unique = true) {
    vector<uint64_t> keys;
    keys.reserve(n);
    if (!unique) {
        for (size_t i = 0; i < n; i++) keys.push_back(rng.next_u64());
        return keys;
    }
    keys.resize(n);
    for (size_t i = 0; i < n; i++) keys[i] = i + 1;
    shuffle(keys.begin(), keys.end(), rng.gen);
    return keys;
}

struct BenchResult {
    size_t n;
    double build_time;
    double query_time;
    double avg_chain = 0.0;
};

BenchResult bench_universal(size_t n, double load_factor = 0.75) {
    RNG rng;
    auto keys = generate_keys(n, rng, true);
    size_t m = (size_t)(n / load_factor);

    double t0 = now_sec();
    UniversalHashTable ht(m, &rng);
    for (auto k : keys) ht.insert(k);
    double t1 = now_sec();

    double t2 = now_sec();
    size_t found = 0;
    for (auto k : keys) if (ht.find(k)) found++;
    double t3 = now_sec();

    return BenchResult{ n, t1 - t0, t3 - t2, ht.avg_chain_length() };
}

const char* py_plot_script = R"PYTHON(
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
)PYTHON";

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    cout << fixed << setprecision(6);

    cout << "=== NASP: Universal Hashing Benchmark ===\n\n";

    vector<size_t> Ns = { 100000, 300000, 1000000, 30000000 };

    ofstream out("hash_benchmarks.csv");
    out << "n,build_time_s,query_time_s,avg_chain\n";

    for (size_t n : Ns) {
        cout << "Testiram n=" << n << "... ";
        auto r = bench_universal(n, 0.75);
        out << r.n << "," << r.build_time << "," << r.query_time << "," << r.avg_chain << "\n";
        cout << "OK (build: " << r.build_time << "s, query: " << r.query_time << "s, avg_chain: " << r.avg_chain << ")\n";
    }

    out.close();
    cout << "CSV snimljen: hash_benchmarks.csv\n\n";

    cout << "Generisem plotove...\n";
    {
        ofstream py("plot_hash.py");
        py << py_plot_script;
    }

    if (system("python plot_hash.py") != 0) {
        cout << "UPOZORENJE: Python nije pronadjen. Pokreni manualno: python plot_hash.py\n";
    }
    else {
        cout << "Python skripta uspjesno pokrenuta.\n";
    }

    cout << "\n=== GOTOVO! ===\n";
    cout << "Datoteke:\n";
    cout << "  - hash_benchmarks.csv\n";
    cout << "  - hash_benchmark_plots.png\n";
    return 0;
}
