// monty_to_compact.cpp
//
// Standalone converter:  Monty "montyformat" *policy* data  ->  compact 16-bit-Gini format.
//
// The Stockfish-side value trainer only uses the policy visit distribution to
// compute one number per position: the Gini impurity (for the "sharp position"
// filter). That distribution is ~82% of the montyformat file. This tool drops it
// and stores the Gini directly as a u16, shrinking the file ~4.6x. On a
// disk-bound training pipeline that is ~4.6x more positions/second.
//
// The source file is only ever READ; the output is a brand-new file.
//
// Build (on the source machine, no dependencies):
//     g++ -O3 -march=native -o monty_to_compact monty_to_compact.cpp
// Run:
//     ./monty_to_compact <in_montyformat.binpack> <out_compact.binpack>
//
// ---------------------------------------------------------------------------
// montyformat (input), little-endian, per game:
//   43-byte header:  4x u64 board, stm/enp/rights/halfm u8, fullm u16,
//                    rf[2][2] u8, result u8
//   per position:    u16 best_move, u16 score, u8 num_moves, num_moves x u8 visits
//   terminator:      u16 best_move == 0
//
// compact (output): 4-byte magic "MGC1", then per game:
//   43-byte header:  copied verbatim
//   per position:    u16 best_move, u16 score, u16 gini_q   (gini_q = round(gini*65535))
//   terminator:      u16 best_move == 0
//
// Gini impurity of the visit distribution v (sum s): 1 - sum((v_i/s)^2).
// gini_q = clamp(round(gini * 65535), 0, 65535); the reader reconstructs
// gini = gini_q / 65535. The trainer only thresholds it (gini < 0.9), so the
// 16-bit quantization is effectively lossless for that decision.
// ---------------------------------------------------------------------------

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <chrono>
#include <sys/stat.h>

static const std::size_t RBUF = std::size_t(64) << 20; // 64 MiB read buffer
static const std::size_t WBUF = std::size_t(64) << 20; // 64 MiB write buffer

struct In {
    std::FILE* f;
    std::vector<unsigned char> b;
    std::size_t pos = 0, len = 0;
    unsigned long long consumed = 0; // bytes parsed (== input offset reached)
    bool eof_ = false;
    explicit In(std::FILE* f_) : f(f_), b(RBUF) {}

    // Make at least `need` bytes available starting at pos. false => clean EOF.
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
    // Return pointer to n bytes and advance. Caller must fill(n) successfully first.
    // Pointer is valid only until the next fill().
    const unsigned char* get(std::size_t n) {
        const unsigned char* p = b.data() + pos;
        pos += n; consumed += n;
        return p;
    }
};

struct Out {
    std::FILE* f;
    std::vector<unsigned char> b;
    std::size_t len = 0;
    unsigned long long written = 0;
    explicit Out(std::FILE* f_) : f(f_), b(WBUF) {}
    void put(const void* p, std::size_t n) {
        if (len + n > b.size()) flush();
        if (n > b.size()) { std::fwrite(p, 1, n, f); written += n; return; }
        std::memcpy(b.data() + len, p, n); len += n;
    }
    void flush() { if (len) { std::fwrite(b.data(), 1, len, f); written += len; len = 0; } }
};

static inline std::uint16_t rd_u16(const unsigned char* p) {
    return static_cast<std::uint16_t>(p[0] | (p[1] << 8));
}
static inline void push_u16(std::vector<unsigned char>& g, std::uint16_t v) {
    g.push_back(static_cast<unsigned char>(v & 0xff));
    g.push_back(static_cast<unsigned char>(v >> 8));
}
static inline double gini_of(const unsigned char* v, int n) {
    long s = 0; for (int i = 0; i < n; ++i) s += v[i];
    if (s <= 0) return 0.0;
    const double inv = 1.0 / static_cast<double>(s);
    double s2 = 0.0;
    for (int i = 0; i < n; ++i) { double p = static_cast<double>(v[i]) * inv; s2 += p * p; }
    return 1.0 - s2;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <in_montyformat.binpack> <out_compact.binpack>\n", argv[0]);
        return 2;
    }
    const char* inpath = argv[1];
    const char* outpath = argv[2];
    if (std::strcmp(inpath, outpath) == 0) { std::fprintf(stderr, "in and out must differ\n"); return 2; }

    std::FILE* fi = std::fopen(inpath, "rb");
    if (!fi) { std::perror("open input"); return 1; }
    std::FILE* fo = std::fopen(outpath, "wb");
    if (!fo) { std::perror("open output"); std::fclose(fi); return 1; }

    struct stat st; unsigned long long insize = 0;
    if (stat(inpath, &st) == 0) insize = static_cast<unsigned long long>(st.st_size);

    In in(fi);
    Out out(fo);
    out.put("MGC1", 4);

    unsigned long long games = 0, positions = 0;
    bool truncated = false;
    auto t0 = std::chrono::steady_clock::now();
    auto tlast = t0;

    std::vector<unsigned char> g; // one game's compact bytes, committed atomically
    g.reserve(4096);

    for (;;) {
        if (!in.fill(43)) break; // clean EOF at a game boundary
        g.clear();
        const unsigned char* hdr = in.get(43);
        g.insert(g.end(), hdr, hdr + 43);

        bool complete = false;
        unsigned long long gpos = 0;
        for (;;) {
            if (!in.fill(2)) break;
            std::uint16_t best = rd_u16(in.get(2));
            if (best == 0) { push_u16(g, 0); complete = true; break; } // terminator
            if (!in.fill(3)) break;                                    // score(2)+num_moves(1)
            const unsigned char* sp = in.get(3);
            std::uint16_t score = rd_u16(sp);
            int nm = sp[2];
            double gini = 0.0;
            if (nm > 0) {
                if (!in.fill(static_cast<std::size_t>(nm))) break;
                gini = gini_of(in.get(nm), nm);
            }
            long q = std::lround(gini * 65535.0);
            if (q < 0) q = 0; if (q > 65535) q = 65535;
            push_u16(g, best);
            push_u16(g, score);
            push_u16(g, static_cast<std::uint16_t>(q));
            ++gpos;
        }
        if (!complete) { truncated = true; break; } // drop the incomplete trailing game

        out.put(g.data(), g.size());
        ++games;
        positions += gpos;

        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration<double>(now - tlast).count() >= 2.0) {
            double el = std::chrono::duration<double>(now - t0).count();
            double rd = static_cast<double>(in.consumed);
            double mbps = el > 0 ? rd / 1e6 / el : 0;
            double frac = insize ? rd / static_cast<double>(insize) : 0;
            double eta = (mbps > 0 && insize) ? (static_cast<double>(insize) - rd) / 1e6 / mbps : 0;
            std::fprintf(stderr,
                "\rgames=%llu pos=%llu  read=%.1f/%.1f GB (%.1f%%)  %.0f MB/s  out=%.1f GB  ETA=%.0f min   ",
                games, positions, rd / 1e9, insize / 1e9, 100 * frac, mbps,
                (out.written + out.len) / 1e9, eta / 60);
            std::fflush(stderr);
            tlast = now;
        }
    }

    out.flush();
    std::fflush(fo);
    bool out_ok = (std::fclose(fo) == 0);
    std::fclose(fi);

    double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    double ratio = out.written ? static_cast<double>(in.consumed) / static_cast<double>(out.written) : 0;
    std::fprintf(stderr,
        "\nDONE  games=%llu positions=%llu  in=%.2f GB  out=%.2f GB  ratio=%.2fx  %.0f s  %s%s\n",
        games, positions, in.consumed / 1e9, out.written / 1e9, ratio, el,
        truncated ? "(stopped at truncated tail) " : "(clean EOF) ",
        out_ok ? "" : "[WARNING: error closing output]");
    return out_ok ? 0 : 1;
}
