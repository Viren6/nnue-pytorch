// monty_to_compact_mt.cpp
//
// Multi-threaded version of monty_to_compact.cpp (montyformat policy -> compact
// 16-bit-Gini). Same output format and same Gini quantization; the single-thread
// converter is CPU-bound scanning the visit distributions, so this fans the Gini
// work out across worker threads to run at disk speed.
//
//   reader thread  : sequential fread + frame whole games -> raw batch queue
//   N worker threads: parse a raw batch, compute Gini per position, emit compact
//   writer thread  : write compact batches to the output file
//
// NOTE: game ORDER is NOT preserved (batches are written as workers finish).
// montyformat games are independent records and the trainer shuffles, so order
// is irrelevant; this is what lets the writer run lock-free of ordering.
//
// The source file is only ever READ; the output is a brand-new file.
//
// Build:  g++ -O3 -march=native -pthread -o monty_to_compact_mt monty_to_compact_mt.cpp
// Run:    ./monty_to_compact_mt <in_montyformat.binpack> <out_compact.binpack> [num_workers]
//
// Output format identical to monty_to_compact.cpp: 4-byte magic "MGC1", then per
// game a 43-byte montyformat header copied verbatim, then fixed 6-byte position
// records (best_move u16, score u16, gini u16 = round(gini*65535)), terminated by
// best_move==0.

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <deque>
#include <thread>
#include <mutex>
#include <atomic>
#include <condition_variable>
#include <chrono>
#include <sys/stat.h>

using Blob = std::vector<unsigned char>;

static const std::size_t RBUF        = std::size_t(64) << 20; // 64 MiB read buffer
static const std::size_t BATCH_BYTES = std::size_t(4)  << 20; // ~4 MiB of whole games per batch

// ------- bounded blocking queue ------------------------------------------
template <class T>
class BQ {
    std::mutex m;
    std::condition_variable not_empty, not_full;
    std::deque<T> q;
    std::size_t cap;
    bool closed = false;
public:
    explicit BQ(std::size_t c) : cap(c) {}
    bool push(T&& v) {
        std::unique_lock<std::mutex> l(m);
        not_full.wait(l, [&] { return q.size() < cap || closed; });
        if (closed) return false;
        q.push_back(std::move(v));
        not_empty.notify_one();
        return true;
    }
    bool pop(T& v) {
        std::unique_lock<std::mutex> l(m);
        not_empty.wait(l, [&] { return !q.empty() || closed; });
        if (q.empty()) return false; // closed + drained
        v = std::move(q.front());
        q.pop_front();
        not_full.notify_one();
        return true;
    }
    void close() {
        std::lock_guard<std::mutex> l(m);
        closed = true;
        not_empty.notify_all();
        not_full.notify_all();
    }
};

// ------- buffered reader (sliding window over fread) ----------------------
struct In {
    std::FILE* f;
    std::vector<unsigned char> b;
    std::size_t pos = 0, len = 0;
    unsigned long long consumed = 0;
    bool eof_ = false;
    explicit In(std::FILE* f_) : f(f_), b(RBUF) {}
    bool fill(std::size_t need) {
        if (len - pos >= need) return true;
        if (pos) { std::memmove(b.data(), b.data() + pos, len - pos); len -= pos; pos = 0; }
        while (len < need && !eof_) {
            std::size_t g = std::fread(b.data() + len, 1, b.size() - len, f);
            if (g == 0) { eof_ = true; break; }
            len += g;
        }
        return len - pos >= need;
    }
    const unsigned char* get(std::size_t n) { const unsigned char* p = b.data() + pos; pos += n; consumed += n; return p; }
};

static inline std::uint16_t rd16(const unsigned char* p) { return std::uint16_t(p[0] | (p[1] << 8)); }
static inline void push16(Blob& o, std::uint16_t v) { o.push_back((unsigned char)(v & 0xff)); o.push_back((unsigned char)(v >> 8)); }
static inline double gini_of(const unsigned char* v, int n) {
    long s = 0; for (int i = 0; i < n; ++i) s += v[i];
    if (s <= 0) return 0.0;
    const double inv = 1.0 / (double)s; double s2 = 0.0;
    for (int i = 0; i < n; ++i) { double p = (double)v[i] * inv; s2 += p * p; }
    return 1.0 - s2;
}

static std::atomic<unsigned long long> g_games{0}, g_positions{0}, g_written{0};
static std::atomic<bool> g_truncated{false};

// convert one raw batch (whole games, montyformat) into a compact batch
static void convert_batch(const Blob& raw, Blob& out) {
    out.clear();
    out.reserve(raw.size() / 4 + 64);
    const unsigned char* p = raw.data();
    const unsigned char* end = p + raw.size();
    unsigned long long pos_count = 0;
    while (p + 43 <= end) {
        out.insert(out.end(), p, p + 43); p += 43;       // header verbatim
        for (;;) {
            if (p + 2 > end) break;
            std::uint16_t best = rd16(p); p += 2;
            if (best == 0) { push16(out, 0); break; }     // terminator
            if (p + 3 > end) break;
            std::uint16_t score = rd16(p); int nm = p[2]; p += 3;
            double gi = 0.0;
            if (nm > 0) { if (p + nm > end) break; gi = gini_of(p, nm); p += nm; }
            long q = std::lround(gi * 65535.0); if (q < 0) q = 0; if (q > 65535) q = 65535;
            push16(out, best); push16(out, score); push16(out, (std::uint16_t)q);
            ++pos_count;
        }
    }
    g_positions.fetch_add(pos_count, std::memory_order_relaxed);
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <in_montyformat.binpack> <out_compact.binpack> [num_workers]\n", argv[0]);
        return 2;
    }
    const char* inpath = argv[1];
    const char* outpath = argv[2];
    if (std::strcmp(inpath, outpath) == 0) { std::fprintf(stderr, "in and out must differ\n"); return 2; }
    int nworkers = (argc > 3) ? std::atoi(argv[3]) : 12;
    if (nworkers < 1) nworkers = 1;

    std::FILE* fi = std::fopen(inpath, "rb");
    if (!fi) { std::perror("open input"); return 1; }
    std::FILE* fo = std::fopen(outpath, "wb");
    if (!fo) { std::perror("open output"); std::fclose(fi); return 1; }

    struct stat st; unsigned long long insize = 0;
    if (stat(inpath, &st) == 0) insize = (unsigned long long)st.st_size;

    // magic first
    std::fwrite("MGC1", 1, 4, fo);
    g_written.store(4);

    BQ<Blob> raw_q(std::size_t(nworkers) * 4 + 4);
    BQ<Blob> out_q(std::size_t(nworkers) * 4 + 4);

    auto t0 = std::chrono::steady_clock::now();

    // writer
    std::thread writer([&] {
        Blob cb;
        while (out_q.pop(cb)) {
            std::fwrite(cb.data(), 1, cb.size(), fo);
            g_written.fetch_add(cb.size(), std::memory_order_relaxed);
        }
    });

    // workers
    std::vector<std::thread> workers;
    for (int i = 0; i < nworkers; ++i) {
        workers.emplace_back([&] {
            Blob raw, out;
            while (raw_q.pop(raw)) {
                convert_batch(raw, out);
                out_q.push(std::move(out));
                out = Blob();
            }
        });
    }

    // reader (this thread)
    {
        In in(fi);
        Blob batch; batch.reserve(BATCH_BYTES + 8192);
        Blob g; g.reserve(8192);
        auto tlast = t0;
        for (;;) {
            if (!in.fill(43)) break; // clean EOF at game boundary
            g.clear();
            const unsigned char* hdr = in.get(43);
            g.insert(g.end(), hdr, hdr + 43);
            bool complete = false;
            for (;;) {
                if (!in.fill(2)) break;
                const unsigned char* mvp = in.get(2);
                g.push_back(mvp[0]); g.push_back(mvp[1]);
                if (mvp[0] == 0 && mvp[1] == 0) { complete = true; break; }
                if (!in.fill(3)) break;
                const unsigned char* sp = in.get(3);
                g.insert(g.end(), sp, sp + 3);
                int nm = sp[2];
                if (nm > 0) {
                    if (!in.fill((std::size_t)nm)) break;
                    const unsigned char* vp = in.get(nm);
                    g.insert(g.end(), vp, vp + nm);
                }
            }
            if (!complete) { g_truncated.store(true); break; }
            batch.insert(batch.end(), g.begin(), g.end());
            g_games.fetch_add(1, std::memory_order_relaxed);
            if (batch.size() >= BATCH_BYTES) {
                raw_q.push(std::move(batch));
                batch = Blob(); batch.reserve(BATCH_BYTES + 8192);
            }
            auto now = std::chrono::steady_clock::now();
            if (std::chrono::duration<double>(now - tlast).count() >= 2.0) {
                double el = std::chrono::duration<double>(now - t0).count();
                double rd = (double)in.consumed, mbps = el > 0 ? rd / 1e6 / el : 0;
                double frac = insize ? rd / (double)insize : 0;
                double eta = (mbps > 0 && insize) ? ((double)insize - rd) / 1e6 / mbps : 0;
                std::fprintf(stderr,
                    "\rgames=%llu pos=%llu  read=%.1f/%.1f GB (%.1f%%)  %.0f MB/s  out=%.1f GB  ETA=%.0f min   ",
                    g_games.load(), g_positions.load(), rd / 1e9, insize / 1e9, 100 * frac, mbps,
                    g_written.load() / 1e9, eta / 60);
                std::fflush(stderr);
                tlast = now;
            }
        }
        if (!batch.empty()) raw_q.push(std::move(batch));
        raw_q.close();
    }

    for (auto& w : workers) w.join();
    out_q.close();
    writer.join();

    std::fflush(fo);
    bool ok = (std::fclose(fo) == 0);
    std::fclose(fi);

    double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    unsigned long long inbytes = 0; { struct stat s2; if (stat(inpath, &s2) == 0) inbytes = s2.st_size; }
    double ratio = g_written.load() ? (double)inbytes / (double)g_written.load() : 0;
    std::fprintf(stderr,
        "\nDONE  games=%llu positions=%llu  in=%.2f GB out=%.2f GB  ratio=%.2fx  %.0f s  workers=%d  %s%s\n",
        g_games.load(), g_positions.load(), inbytes / 1e9, g_written.load() / 1e9, ratio, el, nworkers,
        g_truncated.load() ? "(stopped at truncated tail) " : "(clean EOF) ",
        ok ? "" : "[WARNING: error closing output]");
    return ok ? 0 : 1;
}
