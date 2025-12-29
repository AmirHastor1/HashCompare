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

using u64 = uint64_t;

inline u64 mul_mod(u64 a, u64 b, u64 mod) {
    u64 res = 0;
    a %= mod;
    while (b) {
        if (b & 1) res = (res + a) % mod;
        a = (a * 2) % mod;
        b >>= 1;
    }
    return res;
}

inline u64 hash_mod(u64 a, u64 x, u64 b, u64 p) {
    return (mul_mod(a, x, p) + (b % p)) % p;
}

struct RNG {
    mt19937_64 gen;
    RNG(u64 seed = chrono::high_resolution_clock::now().time_since_epoch().count()) { gen.seed(seed); }
    u64 next_u64() { return gen(); }
    u64 next_u64_range(u64 lo, u64 hi) {
        uniform_int_distribution<u64> dist(lo, hi);
        return dist(gen);
    }
};

struct UniversalHash {
    u64 a, b, p, m;
    UniversalHash() : a(1), b(0), p(1000000007ULL), m(1) {}
    UniversalHash(u64 m_, u64 p_ = 1000000007ULL, RNG* rng = nullptr) : p(p_), m(m_) {
        RNG local;
        if (!rng) rng = &local;
        a = rng->next_u64_range(1, p - 1);
        b = rng->next_u64_range(0, p - 1);
    }
    inline u64 operator()(u64 x) const {
        u64 modp = hash_mod(a, x, b, p);
        return (m == 0) ? 0 : (modp % m);
    }
};

static inline double elapsed_s(chrono::steady_clock::time_point t0, chrono::steady_clock::time_point t1) {
    return chrono::duration<double>(t1 - t0).count();
}

vector<u64> generate_keys(size_t n, RNG& rng, bool unique = true) {
    vector<u64> keys;
    keys.reserve(n);
    if (!unique) {
        for (size_t i = 0; i < n; i++) keys.push_back(rng.next_u64());
        return keys;
    }
    keys.resize(n);
    for (size_t i = 0; i < n; i++) keys[i] = (u64)i + 1; // bitno: 0 koristimo kao EMPTY sentinel
    shuffle(keys.begin(), keys.end(), rng.gen);
    return keys;
}

struct BenchResult {
    string algo;
    size_t n;
    double build_time_s;
    double query_time_s;
    double metric;     // universal: avg_chain, perfect: total_slots/n, cuckoo: rehash_count
};

/* =========================
   1) Universal hashing (chaining)
   ========================= */
struct UniversalHashTable {
    size_t m;
    UniversalHash h;
    vector<vector<u64>> buckets;

    UniversalHashTable(size_t m_, RNG* rng = nullptr)
        : m(m_), h((u64)m_, 1000000007ULL, rng), buckets(m_) {
    }

    void insert(u64 key) {
        u64 idx = h(key);
        auto& v = buckets[idx];
        for (auto x : v) if (x == key) return;
        v.push_back(key);
    }

    bool find(u64 key) const {
        u64 idx = h(key);
        const auto& v = buckets[idx];
        for (auto x : v) if (x == key) return true;
        return false;
    }

    double avg_chain_length() const {
        size_t cnt = 0, used = 0;
        for (const auto& v : buckets) {
            if (!v.empty()) { used++; cnt += v.size(); }
        }
        return used ? (double)cnt / (double)used : 0.0;
    }
};

BenchResult bench_universal(const vector<u64>& keys, double load_factor, RNG& rng) {
    const size_t n = keys.size();
    const size_t m = (size_t)(n / load_factor);

    auto t0 = chrono::steady_clock::now();
    UniversalHashTable ht(m, &rng);
    for (auto k : keys) ht.insert(k);
    auto t1 = chrono::steady_clock::now();

    auto t2 = chrono::steady_clock::now();
    size_t found = 0;
    for (auto k : keys) if (ht.find(k)) found++;
    auto t3 = chrono::steady_clock::now();

    (void)found;
    return BenchResult{ "universal_chain", n, elapsed_s(t0, t1), elapsed_s(t2, t3), ht.avg_chain_length() };
}

/* =========================
   2) Static perfect hashing (FKS 2-level)
   ========================= */
struct PerfectHashTableFKS {
    struct Bucket {
        u64 m2 = 0;
        UniversalHash h2;
        vector<u64> table; // 0 == empty
    };

    u64 m1 = 0;
    UniversalHash h1;
    vector<vector<u64>> bucket_keys;
    vector<Bucket> buckets;
    u64 total_slots = 0;

    void build(const vector<u64>& keys, RNG& rng) {
        const u64 n = (u64)keys.size();
        m1 = max<u64>(1, n);
        bucket_keys.assign((size_t)m1, {});
        buckets.assign((size_t)m1, Bucket{});
        total_slots = 0;

        // Biraj h1 dok suma kvadrata bucket-size-ova ne bude "razumna" (tipično <= 4n)
        // (prakticna FKS heuristika)
        while (true) {
            h1 = UniversalHash(m1, 1000000007ULL, &rng);
            for (auto& v : bucket_keys) v.clear();

            for (u64 k : keys) bucket_keys[(size_t)h1(k)].push_back(k);

            unsigned long long sum_sq = 0;
            for (const auto& v : bucket_keys) {
                unsigned long long b = (unsigned long long)v.size();
                sum_sq += b * b;
            }
            if (sum_sq <= 4ULL * (unsigned long long)n) break;
        }

        // Za svaki bucket: napravi sekundarnu tabelu velicine b^2 i nadji h2 bez kolizija
        for (size_t i = 0; i < (size_t)m1; i++) {
            auto& ks = bucket_keys[i];
            const size_t b = ks.size();
            auto& B = buckets[i];

            if (b == 0) {
                B.m2 = 0;
                B.table.clear();
                continue;
            }

            const u64 m2 = (u64)(b * b);
            B.m2 = m2;
            B.table.assign((size_t)m2, 0);

            while (true) {
                B.h2 = UniversalHash(m2, 1000000007ULL, &rng);
                bool ok = true;
                fill(B.table.begin(), B.table.end(), 0);

                for (u64 k : ks) {
                    const u64 pos = B.h2(k);
                    if (B.table[(size_t)pos] != 0 && B.table[(size_t)pos] != k) {
                        ok = false;
                        break;
                    }
                    B.table[(size_t)pos] = k;
                }
                if (ok) break;
            }

            total_slots += m2;
        }
    }

    bool find(u64 key) const {
        const size_t i = (size_t)h1(key);
        const auto& B = buckets[i];
        if (B.m2 == 0) return false;
        const size_t pos = (size_t)B.h2(key);
        return B.table[pos] == key;
    }
};

BenchResult bench_perfect_fks(const vector<u64>& keys, RNG& rng) {
    const size_t n = keys.size();

    auto t0 = chrono::steady_clock::now();
    PerfectHashTableFKS ht;
    ht.build(keys, rng);
    auto t1 = chrono::steady_clock::now();

    auto t2 = chrono::steady_clock::now();
    size_t found = 0;
    for (auto k : keys) if (ht.find(k)) found++;
    auto t3 = chrono::steady_clock::now();

    (void)found;
    double slots_per_key = (n ? (double)ht.total_slots / (double)n : 0.0); // korisno za uvid u memoriju
    return BenchResult{ "perfect_fks", n, elapsed_s(t0, t1), elapsed_s(t2, t3), slots_per_key };
}

/* =========================
   3) Cuckoo hashing (2 hash funkcije, 1 tabela)
   ========================= */
struct CuckooHashTable {
    u64 m = 0;
    UniversalHash h1, h2;
    vector<u64> table; // 0 == empty
    size_t rehash_count = 0;
    size_t max_kicks = 500;

    CuckooHashTable(u64 m_, RNG& rng, size_t max_kicks_ = 500)
        : m(max<u64>(1, m_)),
        h1(m, 1000000007ULL, &rng),
        h2(m, 1000000007ULL, &rng),
        table((size_t)m, 0),
        max_kicks(max_kicks_) {
    }

    bool find(u64 key) const {
        const size_t p1 = (size_t)h1(key);
        const size_t p2 = (size_t)h2(key);
        return table[p1] == key || table[p2] == key;
    }

    void rebuild_with(const vector<u64>& keys, RNG& rng, u64 new_m) {
        m = max<u64>(1, new_m);
        table.assign((size_t)m, 0);
        h1 = UniversalHash(m, 1000000007ULL, &rng);
        h2 = UniversalHash(m, 1000000007ULL, &rng);

        for (u64 k : keys) {
            bool ok = insert(k, rng);
            if (!ok) {
                // ako zapne i ovdje, samo forsiraj jos vecu tabelu
                rebuild_with(keys, rng, m * 2);
                return;
            }
        }
    }

    bool insert(u64 key, RNG& rng) {
        if (key == 0) return false;
        if (find(key)) return true;

        u64 cur = key;
        size_t pos = (size_t)h1(cur);

        for (size_t kick = 0; kick < max_kicks; kick++) {
            if (table[pos] == 0) {
                table[pos] = cur;
                return true;
            }

            // kick-out
            u64 displaced = table[pos];
            table[pos] = cur;
            cur = displaced;

            // idi na alternativnu poziciju za "cur"
            size_t p1 = (size_t)h1(cur);
            size_t p2 = (size_t)h2(cur);
            pos = (pos == p1) ? p2 : p1;
        }

        // rehash (i po potrebi resize)
        rehash_count++;
        vector<u64> all;
        all.reserve(table.size() + 1);
        for (u64 x : table) if (x != 0) all.push_back(x);
        all.push_back(cur);

        // ako je puno rehash-a, povecaj tabelu (stabilnije za velike n)
        u64 new_m = m;
        if (rehash_count >= 3) new_m = m * 2;

        rebuild_with(all, rng, new_m);
        return true;
    }
};

BenchResult bench_cuckoo(const vector<u64>& keys, double load_factor, RNG& rng) {
    const size_t n = keys.size();
    const u64 m = (u64)max<size_t>(1, (size_t)(n / load_factor));

    auto t0 = chrono::steady_clock::now();
    CuckooHashTable ht(m, rng, 500);

    for (u64 k : keys) ht.insert(k, rng);
    auto t1 = chrono::steady_clock::now();

    auto t2 = chrono::steady_clock::now();
    size_t found = 0;
    for (u64 k : keys) if (ht.find(k)) found++;
    auto t3 = chrono::steady_clock::now();

    (void)found;
    return BenchResult{ "cuckoo", n, elapsed_s(t0, t1), elapsed_s(t2, t3), (double)ht.rehash_count };
}

/* =========================
   Python plot script (3 algos)
   ========================= */
const char* py_log_plot_script = R"PYTHON(
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
)PYTHON";


const char* py_plot_script = R"PYTHON(
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
)PYTHON";

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    cout << fixed << setprecision(6);

    cout << "=== NASP: Hashing Benchmark (Universal vs Perfect(FKS) vs Cuckoo) ===\n\n";

    // NAPOMENA: n=30,000,000 moze biti jako tesko za RAM/CPU (posebno perfect hashing build).
    vector<size_t> Ns = { 80000, 100000, 300000, 500000, 700000, 1000000, 2000000 , 3000000 };

    ofstream out("hash_benchmarks.csv");
    out << "algo,n,build_time_s,query_time_s,metric\n";

    for (size_t n : Ns) {
        RNG rng;
        auto keys = generate_keys(n, rng, true);

        cout << "n=" << n << "\n";

        // 1) Universal chaining (LF 0.75)
        {
            auto r = bench_universal(keys, 0.75, rng);
            out << r.algo << "," << r.n << "," << r.build_time_s << "," << r.query_time_s << "," << r.metric << "\n";
            cout << "  universal_chain: build=" << r.build_time_s << "s, query=" << r.query_time_s
                << "s, avg_chain=" << r.metric << "\n";
        }

        // 2) Perfect hashing FKS (load factor ne postoji isto kao kod tabela; samo build iz skupa kljuceva)
        {
            auto r = bench_perfect_fks(keys, rng);
            out << r.algo << "," << r.n << "," << r.build_time_s << "," << r.query_time_s << "," << r.metric << "\n";
            cout << "  perfect_fks:     build=" << r.build_time_s << "s, query=" << r.query_time_s
                << "s, slots_per_key=" << r.metric << "\n";
        }

        // 3) Cuckoo hashing (preporuka: LF oko 0.45-0.50 za 2-hash cuckoo)
        {
            auto r = bench_cuckoo(keys, 0.45, rng);
            out << r.algo << "," << r.n << "," << r.build_time_s << "," << r.query_time_s << "," << r.metric << "\n";
            cout << "  cuckoo:          build=" << r.build_time_s << "s, query=" << r.query_time_s
                << "s, rehash_count=" << r.metric << "\n";
        }

        cout << "\n";
    }

    out.close();
    cout << "CSV snimljen: hash_benchmarks.csv\n\n";

    cout << "Generisem plotove...\n";
    {
        ofstream py1("plot_hash.py");
        py1 << py_plot_script;

        ofstream py2("log_plot_hash.py");
        py2 << py_log_plot_script;
    } // fajlovi se zatvore ovdje automatski

    if (system("python plot_hash.py") != 0) {
        cout << "UPOZORENJE: Python nije pronadjen. Pokreni manualno: python plot_hash.py\n";
    }
    else {
        cout << "Python skripta (plot_hash.py) uspjesno pokrenuta.\n";
    }

    if (system("python log_plot_hash.py") != 0) {
        cout << "UPOZORENJE: Python nije pronadjen. Pokreni manualno: python log_plot_hash.py\n";
    }
    else {
        cout << "Python skripta (log_plot_hash.py) uspjesno pokrenuta.\n";
    }


    cout << "\n=== GOTOVO! ===\n";
    cout << "Datoteke:\n";
    cout << "  - hash_benchmarks.csv\n";
    cout << "  - hash_benchmark_plots.png\n";
    cout << "  - hash_benchmark_plots_loglog.png\n";

    return 0;
}
