// Debug tool: measure the policy Gini-impurity distribution of a montyformat
// file, pick the median (the threshold that filters ~50% of positions, the
// low-impurity "sharp" half), and report what proportion of CAPTURES and CHECKS
// get filtered at that threshold.
//
//   monty_gini_measure <file> <num_games> [threshold]
//
// Build (from data_loader/cpp/):
//   g++ -O2 -std=c++20 -pthread -I. monty_gini_measure.cpp -o monty_gini_measure
//
// If [threshold] is omitted, the median is used. When [threshold] equals the
// reader's compiled MONTY_GINI_MIN, a cross-check confirms the compiled reader
// drops exactly the same positions.

#include "lib/nnue_training_data_stream.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

using namespace training_data::monty;

template <class T> static bool rd(std::ifstream& f, T& v)
{
    f.read(reinterpret_cast<char*>(&v), sizeof(T));
    return f.gcount() == (std::streamsize)sizeof(T);
}

int main(int argc, char** argv)
{
    if (argc < 3) { std::fprintf(stderr, "usage: %s <file> <num_games> [threshold]\n", argv[0]); return 2; }
    const std::string path = argv[1];
    const int num_games = std::atoi(argv[2]);
    const double thr_arg = (argc > 3) ? std::atof(argv[3]) : -1.0;

    std::ifstream f(path, std::ios::in | std::ios::binary);
    if (!f) { std::fprintf(stderr, "cannot open %s\n", path.c_str()); return 2; }

    std::vector<double> ginis; // double (not float) so the < threshold test matches the reader exactly
    std::vector<char>  is_cap, is_chk;
    // kept (score,ply,result) under the chosen threshold, in stream order
    std::vector<int>   k_score; std::vector<unsigned> k_ply; std::vector<int> k_res;
    long positions = 0; int games = 0;

    for (int g = 0; g < num_games; ++g)
    {
        std::uint64_t qbbs[4]; bool ok = true;
        for (int i = 0; i < 4; ++i) ok = ok && rd(f, qbbs[i]);
        if (!ok) break;
        std::uint8_t stm_u8, enp_u8, rights_u8, halfm_u8; std::uint16_t fullm_u16;
        if (!(rd(f, stm_u8) && rd(f, enp_u8) && rd(f, rights_u8) && rd(f, halfm_u8) && rd(f, fullm_u16))) break;

        MPosition pos;
        decode_quad_bbs(qbbs, pos.bb);
        pos.stm = stm_u8 > 0; pos.enp_sq = enp_u8; pos.rights = rights_u8;
        pos.halfm = halfm_u8; pos.fullm = fullm_u16;

        std::uint8_t rf[2][2];
        for (int s = 0; s < 2; ++s) for (int k = 0; k < 2; ++k) if (!rd(f, rf[s][k])) ok = false;
        if (!ok) break;
        MCastling cast; castling_from_raw(pos, rf, cast);
        std::uint8_t result_u8; if (!rd(f, result_u8)) break;
        const int white_result = (int)result_u8 - 1;

        ++games;
        for (;;)
        {
            std::uint16_t best_move; if (!rd(f, best_move)) { ok = false; break; }
            if (best_move == 0) break;
            std::uint16_t score_raw; std::uint8_t num_moves;
            if (!rd(f, score_raw) || !rd(f, num_moves)) { ok = false; break; }

            double gini = 0.0;
            if (num_moves > 0) {
                std::uint8_t visits[256];
                f.read(reinterpret_cast<char*>(visits), num_moves);
                if (f.gcount() != num_moves) { ok = false; break; }
                gini = policy_gini(visits, num_moves);
            }

            const int  flag = best_move & 15;
            const bool cap  = (flag & 4) != 0;                     // capture / ep / promo-capture
            chess::Position sfp = chess::Position::fromFen(pos.sf_fen().c_str());
            const bool chk  = sfp.isCheck();                       // side to move in check

            ginis.push_back(gini); is_cap.push_back(cap); is_chk.push_back(chk);

            const int stmv = pos.stm ? 1 : 0;
            k_score.push_back((int)score_to_cp((float)score_raw / (float)0xFFFF));
            k_ply.push_back((unsigned)(2 * ((int)pos.fullm - 1) + stmv));
            k_res.push_back(stmv ? -white_result : white_result);
            ++positions;

            pos.make(best_move, cast);
        }
        if (!ok) break;
    }

    // median Gini
    std::vector<double> sorted = ginis;
    std::sort(sorted.begin(), sorted.end());
    const double median = sorted.empty() ? 0.0 : sorted[sorted.size() / 2];
    const double threshold = (thr_arg >= 0.0) ? thr_arg : median;

    // The actual training filter drops a position if its policy is sharp
    // (gini < threshold) OR the best move is a capture OR it is in check.
    long ng = 0;                         // gini-only filtered
    long ncap = 0, ncap_gf = 0;          // captures, and how many gini alone drops
    long nchk = 0, nchk_gf = 0;          // checks,   and how many gini alone drops
    long ncomb = 0;                      // combined-filtered
    std::vector<int> ks2; std::vector<unsigned> kp2; std::vector<int> kr2; // KEPT under combined filter (stream order)
    for (long i = 0; i < positions; ++i)
    {
        const bool gfilt = ginis[i] < threshold;
        const bool comb  = gfilt || is_cap[i] || is_chk[i];
        if (gfilt) ++ng;
        if (comb)  ++ncomb;
        if (is_cap[i]) { ++ncap; if (gfilt) ++ncap_gf; }
        if (is_chk[i]) { ++nchk; if (gfilt) ++nchk_gf; }
        if (!comb) { ks2.push_back(k_score[i]); kp2.push_back(k_ply[i]); kr2.push_back(k_res[i]); }
    }

    auto pct = [](long a, long b){ return b ? 100.0 * a / b : 0.0; };
    std::printf("games=%d positions=%ld\n", games, positions);
    std::printf("Gini: median=%.5f  (p10=%.3f p25=%.3f p50=%.3f p75=%.3f p90=%.3f)\n",
                median,
                sorted[sorted.size()/10], sorted[sorted.size()/4], sorted[sorted.size()/2],
                sorted[sorted.size()*3/4], sorted[sorted.size()*9/10]);
    std::printf("threshold=%.5f (compiled MONTY_GINI_MIN=%.5f)\n", threshold, MONTY_GINI_MIN);
    std::printf("-- Gini test alone --\n");
    std::printf("   overall filtered: %ld / %ld (%.2f%%)\n", ng, positions, pct(ng, positions));
    std::printf("   captures: %ld (%.2f%% of data); gini drops %ld (%.2f%%)\n",
                ncap, pct(ncap, positions), ncap_gf, pct(ncap_gf, ncap));
    std::printf("   checks:   %ld (%.2f%% of data); gini drops %ld (%.2f%%)\n",
                nchk, pct(nchk, positions), nchk_gf, pct(nchk_gf, nchk));
    std::printf("-- Combined filter (gini OR capture OR check) = actual training --\n");
    std::printf("   overall filtered: %ld / %ld (%.2f%%)\n", ncomb, positions, pct(ncomb, positions));
    std::printf("   captures filtered: %.2f%%   checks filtered: %.2f%%  (both guaranteed 100%%)\n",
                pct(ncap, ncap), pct(nchk, nchk));

    // cross-check: the compiled reader must keep exactly the combined-filter set.
    {
        MontyFenInputStream st(path, false, nullptr, 0, 1);
        long cmp = (long)ks2.size(); if (cmp > 20000) cmp = 20000;
        long mism = 0, pulled = 0;
        for (long i = 0; i < cmp; ++i)
        {
            auto e = st.next();
            if (!e.has_value()) break;
            ++pulled;
            if (e->score != ks2[i] || e->ply != kp2[i] || e->result != kr2[i]) ++mism;
        }
        const bool thr_matches = std::abs((double)MONTY_GINI_MIN - threshold) < 1e-9;
        std::printf("reader cross-check: compared %ld, mismatches %ld %s\n",
                    pulled, mism,
                    (mism == 0 && thr_matches) ? "(reader == combined filter OK)"
                    : "(run with threshold == compiled MONTY_GINI_MIN to validate)");
    }
    return 0;
}
