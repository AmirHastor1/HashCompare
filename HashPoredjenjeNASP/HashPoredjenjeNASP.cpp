#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <random>
#include <algorithm>
#include <fstream>
#include <functional>
#include <iomanip>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <intrin.h>

using namespace std;
using u64 = uint64_t;

/* =========================
   Fast mod 2^61-1 multiply
   ========================= */
static constexpr u64 P61 = (1ULL << 61) - 1;

static inline u64 modp61(u64 x) {
    x = (x & P61) + (x >> 61);
    if (x >= P61) x -= P61;
    return x;
}

static inline u64 mul_mod_p61(u64 a, u64 b) {
#if defined(_MSC_VER) && defined(_M_X64)
    unsigned __int64 hi;
    unsigned __int64 lo = _umul128((unsigned __int64)a, (unsigned __int64)b, &hi); // x64 intrinsic [page:0]
    u64 low_masked = (u64)lo & P61;
    u64 shifted = ((u64)hi << 3) | ((u64)lo >> 61);
    u64 res = low_masked + shifted;
    res = modp61(res);
    return res;
#else
    unsigned __int128 z = (unsigned __int128)a * (unsigned __int128)b;
    u64 low = (u64)z & P61;
    u64 high = (u64)(z >> 61);
    u64 res = low + high;
    res = modp61(res);
    return res;
#endif
}

struct RNG {
    mt19937_64 gen;
    RNG(u64 seed = chrono::high_resolution_clock::now().time_since_epoch().count()) { gen.seed(seed); }
    u64 next_u64() { return gen(); }
    u64 next_u64_range(u64 lo, u64 hi) {
        uniform_int_distribution<u64> dist(lo, hi);
        return dist(gen);
    }
    int next_int_range(int lo, int hi) {
        uniform_int_distribution<int> dist(lo, hi);
        return dist(gen);
    }
};

struct UniversalHash64 {
    u64 a = 1, b = 0, m = 1;
    UniversalHash64() = default;
    UniversalHash64(u64 m_, RNG& rng) : m(max<u64>(1, m_)) {
        a = rng.next_u64_range(1, P61 - 1);
        b = rng.next_u64_range(0, P61 - 1);
    }
    inline u64 operator()(u64 x) const {
        u64 y = modp61(mul_mod_p61(a, modp61(x)) + b);
        return (m == 0) ? 0 : (y % m);
    }
};

static inline double elapsed_s(chrono::steady_clock::time_point t0, chrono::steady_clock::time_point t1) {
    return chrono::duration<double>(t1 - t0).count();
}

/* =========================
   Key -> u64
   ========================= */
static inline u64 splitmix64(u64 x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

static inline u64 hash_bytes_fnv1a(const void* data, size_t len) {
    const unsigned char* p = (const unsigned char*)data;
    u64 h = 1469598103934665603ULL;
    for (size_t i = 0; i < len; i++) {
        h ^= (u64)p[i];
        h *= 1099511628211ULL;
    }
    return h;
}

template <class Key>
struct KeyToU64;

template <>
struct KeyToU64<u64> {
    static inline u64 conv(u64 x) { return splitmix64(x); }
};

template <>
struct KeyToU64<string> {
    static inline u64 conv(const string& s) {
        return splitmix64(hash_bytes_fnv1a(s.data(), s.size()));
    }
};

template <>
struct KeyToU64<pair<u64, u64>> {
    static inline u64 conv(const pair<u64, u64>& p) {
        u64 x = splitmix64(p.first);
        u64 y = splitmix64(p.second);
        return splitmix64(x ^ (y + 0x9e3779b97f4a7c15ULL + (x << 6) + (x >> 2)));
    }
};

// NEW: pair<string,string>
template <>
struct KeyToU64<pair<string, string>> {
    static inline u64 conv(const pair<string, string>& p) {
        u64 h1 = KeyToU64<string>::conv(p.first);
        u64 h2 = KeyToU64<string>::conv(p.second);
        // simple combine then mix
        return splitmix64(h1 ^ (h2 + 0x9e3779b97f4a7c15ULL + (h1 << 6) + (h1 >> 2)));
    }
};

/* =========================
   Generators
   ========================= */
vector<u64> generate_keys_u64_unique(size_t n, RNG& rng) {
    vector<u64> keys(n);
    for (size_t i = 0; i < n; i++) keys[i] = (u64)i + 1;
    shuffle(keys.begin(), keys.end(), rng.gen);
    return keys;
}

vector<string> generate_keys_string_unique(size_t n, RNG& rng, int len = 20) {
    vector<string> keys;
    keys.reserve(n);
    const string alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    for (size_t i = 0; i < n; i++) {
        string s;
        s.reserve((size_t)len + 24);
        s += to_string(i + 1);
        s.push_back('_');
        while ((int)s.size() < len) s.push_back(alphabet[(size_t)rng.next_int_range(0, (int)alphabet.size() - 1)]);
        keys.push_back(std::move(s));
    }
    shuffle(keys.begin(), keys.end(), rng.gen);
    return keys;
}

vector<pair<u64, u64>> generate_keys_pair_unique(size_t n, RNG& rng) {
    vector<pair<u64, u64>> keys;
    keys.reserve(n);
    for (size_t i = 0; i < n; i++) keys.push_back({ (u64)i + 1, rng.next_u64() });
    shuffle(keys.begin(), keys.end(), rng.gen);
    return keys;
}

/* =========================
   Dataset loader -> pair<string,string> (first 2 columns)
   ========================= */
static inline void rtrim_newlines(string& s) {
    while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) s.pop_back();
}

static inline bool parse_first_two_csv_cols_simple(const string& line, string& c1, string& c2) {
    // Simple CSV split by commas: c1 = before first comma, c2 = between first and second (or rest) [web:180]
    size_t p1 = line.find(',');
    if (p1 == string::npos) return false;
    size_t p2 = line.find(',', p1 + 1);

    c1 = line.substr(0, p1);
    if (p2 == string::npos) c2 = line.substr(p1 + 1);
    else c2 = line.substr(p1 + 1, p2 - (p1 + 1));

    return (!c1.empty() && !c2.empty());
}

vector<pair<string, string>> load_pair_dataset_first2cols(const string& path, size_t max_keys) {
    ifstream in(path);
    if (!in) throw runtime_error("Ne mogu otvoriti dataset: " + path);

    vector<pair<string, string>> keys;
    keys.reserve(max_keys ? max_keys : 1024);

    string line;
    while (std::getline(in, line)) {
        rtrim_newlines(line);
        if (line.empty()) continue;

        string c1, c2;
        if (!parse_first_two_csv_cols_simple(line, c1, c2)) continue;

        keys.push_back({ std::move(c1), std::move(c2) });
        if (max_keys && keys.size() >= max_keys) break;
    }
    return keys;
}

/* =========================
   Benchmark result
   ========================= */
struct BenchResult {
    string algo;
    string dtype;
    size_t n;
    double fill_time_s;
    double query_time_s;
};

/* =========================
   1) Universal chaining
   ========================= */
template <class Key>
struct UniversalHashTable {
    size_t m;
    UniversalHash64 h;
    vector<vector<u64>> buckets;

    UniversalHashTable(size_t m_, RNG& rng)
        : m(max<size_t>(1, m_)), h((u64)m, rng), buckets(m) {}

    void insert_u64(u64 ku) {
        u64 idx = h(ku);
        auto& v = buckets[(size_t)idx];
        for (u64 x : v) if (x == ku) return;
        v.push_back(ku);
    }

    bool find_u64(u64 ku) const {
        u64 idx = h(ku);
        const auto& v = buckets[(size_t)idx];
        for (u64 x : v) if (x == ku) return true;
        return false;
    }
};

template <class Key>
BenchResult bench_universal(const vector<Key>& keys, double load_factor, RNG& rng, const string& dtype) {
    const size_t n = keys.size();
    const size_t m = max<size_t>(1, (size_t)(n / load_factor));
    UniversalHashTable<Key> ht(m, rng);

    auto tf0 = chrono::steady_clock::now();
    for (const auto& k : keys) ht.insert_u64(KeyToU64<Key>::conv(k));
    auto tf1 = chrono::steady_clock::now();

    auto tq0 = chrono::steady_clock::now();
    size_t found = 0;
    for (const auto& k : keys) if (ht.find_u64(KeyToU64<Key>::conv(k))) found++;
    auto tq1 = chrono::steady_clock::now();
    (void)found;

    return BenchResult{ "universal_chain", dtype, n, elapsed_s(tf0, tf1), elapsed_s(tq0, tq1) };
}

/* =========================
   2) Perfect hashing FKS static
   ========================= */
struct PerfectHashTableFKS {
    struct Bucket {
        u64 m2 = 0;
        UniversalHash64 h2;
        vector<u64> table;
    };

    u64 m1 = 0;
    UniversalHash64 h1;
    vector<vector<u64>> bucket_keys;
    vector<Bucket> buckets;

    void build_from_u64(const vector<u64>& keys_u64, RNG& rng) {
        const u64 n = (u64)keys_u64.size();
        m1 = max<u64>(1, n);
        bucket_keys.assign((size_t)m1, {});
        buckets.assign((size_t)m1, Bucket{});

        while (true) {
            h1 = UniversalHash64(m1, rng);
            for (auto& v : bucket_keys) v.clear();
            for (u64 ku : keys_u64) bucket_keys[(size_t)h1(ku)].push_back(ku);

            unsigned long long sum_sq = 0;
            for (const auto& v : bucket_keys) {
                unsigned long long b = (unsigned long long)v.size();
                sum_sq += b * b;
            }
            if (sum_sq <= 4ULL * (unsigned long long)n) break;
        }

        for (size_t i = 0; i < (size_t)m1; i++) {
            auto& ks = bucket_keys[i];
            const size_t b = ks.size();
            auto& B = buckets[i];

            if (b == 0) { B.m2 = 0; B.table.clear(); continue; }

            const u64 m2 = (u64)(b * b);
            B.m2 = m2;
            B.table.assign((size_t)m2, 0);

            while (true) {
                B.h2 = UniversalHash64(m2, rng);
                bool ok = true;
                fill(B.table.begin(), B.table.end(), 0);

                for (u64 ku : ks) {
                    size_t pos = (size_t)B.h2(ku);
                    u64& cell = B.table[pos];
                    if (cell != 0 && cell != ku) { ok = false; break; }
                    cell = ku;
                }
                if (ok) break;
            }
        }
    }

    bool find_u64(u64 ku) const {
        size_t i = (size_t)h1(ku);
        const auto& B = buckets[i];
        if (B.m2 == 0) return false;
        size_t pos = (size_t)B.h2(ku);
        return B.table[pos] == ku;
    }
};

template <class Key>
BenchResult bench_perfect_fks_static(const vector<Key>& keys, RNG& rng, const string& dtype) {
    const size_t n = keys.size();
    vector<u64> keys_u64;
    keys_u64.reserve(n);
    for (const auto& k : keys) {
        u64 ku = KeyToU64<Key>::conv(k);
        if (ku == 0) ku = 1;
        keys_u64.push_back(ku);
    }

    auto tf0 = chrono::steady_clock::now();
    PerfectHashTableFKS ht;
    ht.build_from_u64(keys_u64, rng);
    auto tf1 = chrono::steady_clock::now();

    auto tq0 = chrono::steady_clock::now();
    size_t found = 0;
    for (u64 ku : keys_u64) if (ht.find_u64(ku)) found++;
    auto tq1 = chrono::steady_clock::now();
    (void)found;

    return BenchResult{ "perfect_fks_static", dtype, n, elapsed_s(tf0, tf1), elapsed_s(tq0, tq1) };
}

/* =========================
   2b) Perfect dynamic bucket rebuild
   ========================= */
struct PerfectHashDynamicBuckets {
    struct Bucket {
        vector<u64> keys;
        u64 m2 = 0;
        UniversalHash64 h2;
        vector<u64> table;
    };

    u64 m1 = 0;
    UniversalHash64 h1;
    vector<Bucket> buckets;

    void init(u64 expected_n, RNG& rng) {
        m1 = max<u64>(1, expected_n);
        h1 = UniversalHash64(m1, rng);
        buckets.assign((size_t)m1, Bucket{});
    }

    void rebuild_bucket(size_t i, RNG& rng) {
        auto& B = buckets[i];
        const size_t b = B.keys.size();
        if (b == 0) { B.m2 = 0; B.table.clear(); return; }

        u64 m2 = (u64)(b * b);
        B.m2 = m2;
        B.table.assign((size_t)m2, 0);

        while (true) {
            B.h2 = UniversalHash64(m2, rng);
            fill(B.table.begin(), B.table.end(), 0);

            bool ok = true;
            for (u64 ku : B.keys) {
                size_t pos = (size_t)B.h2(ku);
                u64& cell = B.table[pos];
                if (cell != 0 && cell != ku) { ok = false; break; }
                cell = ku;
            }
            if (ok) break;
        }
    }

    bool find_u64(u64 ku) const {
        size_t i = (size_t)h1(ku);
        const auto& B = buckets[i];
        if (B.m2 == 0) return false;
        size_t pos = (size_t)B.h2(ku);
        return B.table[pos] == ku;
    }

    void insert_u64(u64 ku, RNG& rng) {
        size_t i = (size_t)h1(ku);
        auto& B = buckets[i];

        if (B.m2 != 0) {
            size_t pos = (size_t)B.h2(ku);
            u64 cell = B.table[pos];
            if (cell == ku) return;
            if (cell == 0) {
                B.table[pos] = ku;
                B.keys.push_back(ku);
                return;
            }
        }
        B.keys.push_back(ku);
        rebuild_bucket(i, rng);
    }
};

template <class Key>
BenchResult bench_perfect_dynamic_bucket(const vector<Key>& keys, RNG& rng, const string& dtype) {
    const size_t n = keys.size();
    vector<u64> keys_u64;
    keys_u64.reserve(n);
    for (const auto& k : keys) {
        u64 ku = KeyToU64<Key>::conv(k);
        if (ku == 0) ku = 1;
        keys_u64.push_back(ku);
    }

    PerfectHashDynamicBuckets ht;
    ht.init((u64)n, rng);

    auto tf0 = chrono::steady_clock::now();
    for (u64 ku : keys_u64) ht.insert_u64(ku, rng);
    auto tf1 = chrono::steady_clock::now();

    auto tq0 = chrono::steady_clock::now();
    size_t found = 0;
    for (u64 ku : keys_u64) if (ht.find_u64(ku)) found++;
    auto tq1 = chrono::steady_clock::now();
    (void)found;

    return BenchResult{ "perfect_dynamic_bucket", dtype, n, elapsed_s(tf0, tf1), elapsed_s(tq0, tq1) };
}

/* =========================
   3) Cuckoo hashing
   ========================= */
struct CuckooHashTable {
    u64 m = 0;
    UniversalHash64 h1, h2;
    vector<u64> table;
    size_t rehash_count = 0;
    size_t max_kicks = 500;

    CuckooHashTable(u64 m_, RNG& rng, size_t max_kicks_ = 500)
        : m(max<u64>(1, m_)),
        h1(m, rng),
        h2(m, rng),
        table((size_t)m, 0),
        max_kicks(max_kicks_) {}

    bool find(u64 ku) const {
        size_t p1 = (size_t)h1(ku);
        size_t p2 = (size_t)h2(ku);
        return table[p1] == ku || table[p2] == ku;
    }

    bool insert(u64 ku, RNG& rng) {
        if (ku == 0) return false;
        if (find(ku)) return true;

        u64 cur = ku;
        size_t pos = (size_t)h1(cur);

        for (size_t kick = 0; kick < max_kicks; kick++) {
            if (table[pos] == 0) { table[pos] = cur; return true; }

            u64 displaced = table[pos];
            table[pos] = cur;
            cur = displaced;

            size_t p1 = (size_t)h1(cur);
            size_t p2 = (size_t)h2(cur);
            pos = (pos == p1) ? p2 : p1;
        }

        rehash_count++;
        vector<u64> all;
        all.reserve(table.size() + 1);
        for (u64 x : table) if (x != 0) all.push_back(x);
        all.push_back(cur);

        u64 new_m = m;
        if (rehash_count >= 3) new_m = m * 2;

        rebuild_with(all, rng, new_m);
        return true;
    }

    void rebuild_with(const vector<u64>& keys, RNG& rng, u64 new_m) {
        m = max<u64>(1, new_m);
        table.assign((size_t)m, 0);
        h1 = UniversalHash64(m, rng);
        h2 = UniversalHash64(m, rng);

        rehash_count = 0;
        for (u64 ku : keys) {
            bool ok = insert(ku, rng);
            if (!ok) {
                rebuild_with(keys, rng, m * 2);
                return;
            }
        }
    }
};

template <class Key>
BenchResult bench_cuckoo(const vector<Key>& keys, double load_factor, RNG& rng, const string& dtype) {
    const size_t n = keys.size();
    vector<u64> keys_u64;
    keys_u64.reserve(n);
    for (const auto& k : keys) {
        u64 ku = KeyToU64<Key>::conv(k);
        if (ku == 0) ku = 1;
        keys_u64.push_back(ku);
    }

    const u64 m = (u64)max<size_t>(1, (size_t)(n / load_factor));
    CuckooHashTable ht(m, rng, 500);

    auto tf0 = chrono::steady_clock::now();
    for (u64 ku : keys_u64) ht.insert(ku, rng);
    auto tf1 = chrono::steady_clock::now();

    auto tq0 = chrono::steady_clock::now();
    size_t found = 0;
    for (u64 ku : keys_u64) if (ht.find(ku)) found++;
    auto tq1 = chrono::steady_clock::now();
    (void)found;

    return BenchResult{ "cuckoo", dtype, n, elapsed_s(tf0, tf1), elapsed_s(tq0, tq1) };
}

/* =========================
   Python plotting script (unchanged)
   ========================= */
static const char* py_plot_all_script = R"PY(
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
)PY";

/* =========================
   CLI args
   ========================= */
struct CmdArgs {
    string dataset_path;
    size_t dataset_max = 0;
};

CmdArgs parse_args(int argc, char** argv) {
    CmdArgs a;
    for (int i = 1; i < argc; i++) {
        string arg = argv[i];
        if (arg == "--dataset" && i + 1 < argc) a.dataset_path = argv[++i];
        else if (arg == "--max" && i + 1 < argc) a.dataset_max = (size_t)stoull(argv[++i]);
    }
    return a;
}

/* =========================
   Run suite -> one CSV
   ========================= */
template <class Key>
void run_one_dtype(const string& dtype,
    const vector<size_t>& Ns,
    function<vector<Key>(size_t, RNG&)> gen_keys,
    ofstream& out) {
    for (size_t n : Ns) {
        RNG rng;
        auto keys = gen_keys(n, rng);

        cout << "dtype=" << dtype << " n=" << n << "\n";

        auto r1 = bench_universal<Key>(keys, 0.75, rng, dtype);
        out << r1.algo << "," << r1.dtype << "," << r1.n << "," << r1.fill_time_s << "," << r1.query_time_s << "\n";

        auto r2 = bench_perfect_fks_static<Key>(keys, rng, dtype);
        out << r2.algo << "," << r2.dtype << "," << r2.n << "," << r2.fill_time_s << "," << r2.query_time_s << "\n";

        auto r3 = bench_perfect_dynamic_bucket<Key>(keys, rng, dtype);
        out << r3.algo << "," << r3.dtype << "," << r3.n << "," << r3.fill_time_s << "," << r3.query_time_s << "\n";

        auto r4 = bench_cuckoo<Key>(keys, 0.45, rng, dtype);
        out << r4.algo << "," << r4.dtype << "," << r4.n << "," << r4.fill_time_s << "," << r4.query_time_s << "\n";

        cout << "\n";
    }
}

int main(int argc, char** argv) {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    cout << fixed << setprecision(6);

    CmdArgs args = parse_args(argc, argv);

    cout << "=== NASP: Hashing Benchmark (fill + query) ===\n";

    vector<size_t> Ns = { 100000, 300000, 500000, 700000, 1000000, 3000000, 5000000, 7000000 };

    ofstream out("hash_bench_all.csv");
    out << "algo,dtype,n,fill_time_s,query_time_s\n";

    run_one_dtype<u64>("u64", Ns,
        [](size_t n, RNG& rng) { return generate_keys_u64_unique(n, rng); }, out);

    run_one_dtype<string>("string", Ns,
        [](size_t n, RNG& rng) { return generate_keys_string_unique(n, rng, 20); }, out);

    run_one_dtype<pair<u64, u64>>("pair", Ns,
        [](size_t n, RNG& rng) { return generate_keys_pair_unique(n, rng); }, out);

    // dataset-based test as pair<string,string>
    if (!args.dataset_path.empty()) {
        cout << "Ucitam dataset (pair): " << args.dataset_path << "\n";
        auto data = load_pair_dataset_first2cols(args.dataset_path, args.dataset_max);
        cout << "Dataset pairs loaded: " << data.size() << "\n";

        vector<size_t> Ns_ds;
        size_t M = data.size();
        vector<size_t> candidates = { 50000, 100000, 200000, 300000, 500000, 700000, 900000, 1000000 };
        for (size_t x : candidates) if (x <= M) Ns_ds.push_back(x);
        if (Ns_ds.empty()) Ns_ds.push_back(M);

        for (size_t n : Ns_ds) {
            vector<pair<string, string>> prefix(data.begin(), data.begin() + n);
            RNG rng;
            cout << "dtype=pair_dataset n=" << n << "\n";

            auto r1 = bench_universal<pair<string, string>>(prefix, 0.75, rng, "pair_dataset");
            out << r1.algo << "," << r1.dtype << "," << r1.n << "," << r1.fill_time_s << "," << r1.query_time_s << "\n";

            auto r2 = bench_perfect_fks_static<pair<string, string>>(prefix, rng, "pair_dataset");
            out << r2.algo << "," << r2.dtype << "," << r2.n << "," << r2.fill_time_s << "," << r2.query_time_s << "\n";

            auto r3 = bench_perfect_dynamic_bucket<pair<string, string>>(prefix, rng, "pair_dataset");
            out << r3.algo << "," << r3.dtype << "," << r3.n << "," << r3.fill_time_s << "," << r3.query_time_s << "\n";

            auto r4 = bench_cuckoo<pair<string, string>>(prefix, 0.45, rng, "pair_dataset");
            out << r4.algo << "," << r4.dtype << "," << r4.n << "," << r4.fill_time_s << "," << r4.query_time_s << "\n";

            cout << "\n";
        }
    }

    out.close();
    cout << "CSV snimljen: hash_bench_all.csv\n";

    {
        ofstream py("plot_all.py");
        py << py_plot_all_script;
    }
    if (system("python plot_all.py") != 0) {
        cout << "UPOZORENJE: Python nije pronadjen. Pokreni manualno: python plot_all.py\n";
    }

    return 0;
}
