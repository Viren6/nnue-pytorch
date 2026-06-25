#ifndef _MONTY_STREAM_H_
#define _MONTY_STREAM_H_

// Reader for Monty's "montyformat" *policy* data files (e.g. produced by
// monty/crates/datagen with --policy-data, then interleaved).
//
// Monty stores, per game, a start position + a list of (best_move, win-prob
// score, visit distribution).  The Stockfish NNUE trainer is a *value* trainer,
// so this reader extracts, for every position, exactly what binpack::
// TrainingDataEntry needs and DROPS the policy visit distribution (after using
// it for the Gini filter):
//
//   pos    : reconstructed by replaying best moves from the start position
//   score  : win-prob (side-to-move relative, [0,1]) -> stm centipawns
//   result : Monty stores game result white-relative; flipped to stm-relative
//   ply    : 2*(fullm-1) + stm
//   move   : best_move (best-effort; only used by optional "filtered" skipping)
//
// Filtering (see read/decode): a position is dropped if its policy is "sharp"
// (Gini impurity < MONTY_GINI_MIN), the best move is a capture, or the side to
// move is in check; the trainer's skipPredicate is then applied on top.
//
// THREADING: decoding (replay + fromFen + isCheck) is CPU-heavy, so it is done
// in parallel across `concurrency` producer threads. Each producer grabs raw
// game bytes under a short lock (cheap I/O) and decodes them lock-free, pushing
// decoded chunks into a bounded queue that fill()/next() drain. This keeps the
// GPU fed; a single-threaded decoder cannot. With concurrency==1 the output is
// in deterministic file order (used by tools and the verifier).
//
// Conventions are taken verbatim from monty/crates/montyformat
// (format.rs, value.rs, chess/{position,moves,frc,consts}.rs) and verified
// position-by-position against Monty's own decoder. The on-disk format is
// little-endian and uncompressed; this reader assumes a little-endian host.

#include "training_data_entry.h" // binpack::TrainingDataEntry + chess::Position
#include "thread_safe_types.h"   // ThreadSafeRingBuffer

#include <array>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <deque>
#include <fstream>
#include <functional>
#include <mutex>
#include <optional>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace training_data {
namespace monty {

    using namespace binpack; // for TrainingDataEntry (as in nnue_training_data_stream.h)

    // Monty piece indices: 2=Pawn .. 7=King ; side 0=White, 1=Black.
    // Square index a1=0 .. h8=63 (matches Stockfish chess::Square ordinal).

    inline int mctz64(std::uint64_t b) { return __builtin_ctzll(b); }

    struct MCastling
    {
        bool          chess960 = false;
        std::uint8_t  castle_mask[64];
        std::uint8_t  rook_files[2][2]; // [side][ks] ks: 0=queenside, 1=kingside

        std::uint8_t mask(int sq) const { return castle_mask[sq]; }
        int rook_file(int side, int ks) const { return rook_files[side][ks]; }
    };

    struct MPosition
    {
        std::uint64_t bb[8];     // 0=W,1=B,2=P,3=N,4=B,5=R,6=Q,7=K
        bool          stm;       // false=white, true=black
        std::uint8_t  enp_sq;
        std::uint8_t  rights;    // WQS=8 WKS=4 BQS=2 BKS=1
        std::uint8_t  halfm;
        std::uint16_t fullm;

        int get_pc(std::uint64_t bit) const
        {
            for (int pc = 2; pc <= 7; ++pc)
                if (bit & bb[pc]) return pc;
            return 0;
        }

        void toggle(int side, int pc, int sq)
        {
            std::uint64_t b = 1ull << sq;
            bb[pc] ^= b;
            bb[side] ^= b;
        }

        int king_sq(int side) const { return mctz64(bb[7] & bb[side]); }

        // Faithful port of montyformat chess::Position::make.
        void make(std::uint16_t mov, const MCastling& c)
        {
            const int side   = stm ? 1 : 0;
            const int from   = mov >> 10;
            const int to     = (mov >> 4) & 63;
            const int flag   = mov & 15;
            const std::uint64_t bb_to = 1ull << to;
            const int moved    = get_pc(1ull << from);
            const bool is_cap  = (flag & 4) != 0;            // CAP bit
            const int captured = is_cap ? get_pc(bb_to) : 0; // 0 for en-passant target

            stm    = !stm;
            enp_sq = 0;
            rights &= c.mask(to) & c.mask(from);
            halfm  += 1;
            fullm  += (side == 1) ? 1 : 0;                   // increment after black moves
            if (moved == 2 /*pawn*/ || is_cap) halfm = 0;

            toggle(side, moved, from);
            toggle(side, moved, to);
            if (captured != 0) toggle(side ^ 1, captured, to);

            switch (flag)
            {
                case 1: // DBL
                    enp_sq = static_cast<std::uint8_t>(to ^ 8);
                    break;
                case 2: // KS
                case 3: // QS
                {
                    const int ks  = (flag == 2) ? 1 : 0;     // KS -> 1, QS -> 0
                    const int sf  = 56 * side;
                    const int rfr = sf + c.rook_file(side, ks);
                    const int rto = sf + (ks ? 5 : 3);       // [3,5][ks]
                    toggle(side, 5 /*rook*/, rfr);
                    toggle(side, 5 /*rook*/, rto);
                    break;
                }
                case 5: // ENP
                    toggle(side ^ 1, 2 /*pawn*/, to ^ 8);
                    break;
                default:
                    if (flag >= 8) // promotions NPR.. QPC
                    {
                        const int promo = (flag & 3) + 3;
                        toggle(side, 2 /*pawn*/, to);
                        toggle(side, promo, to);
                    }
                    break;
            }
        }

        void append_placement(std::string& fen) const
        {
            static const char PIECES[12] =
                {'P','N','B','R','Q','K','p','n','b','r','q','k'};
            for (int rank = 7; rank >= 0; --rank)
            {
                int clear = 0;
                for (int file = 0; file < 8; ++file)
                {
                    const int sq = 8 * rank + file;
                    const std::uint64_t bit = 1ull << sq;
                    const int pc = get_pc(bit);
                    if (pc != 0)
                    {
                        if (clear > 0) fen += std::to_string(clear);
                        clear = 0;
                        const int black = (bb[1] & bit) ? 1 : 0;
                        fen += PIECES[pc - 2 + 6 * black];
                    }
                    else
                    {
                        ++clear;
                    }
                }
                if (clear > 0) fen += std::to_string(clear);
                if (rank > 0) fen += '/';
            }
        }

        // Mimics Monty's Position::as_fen EXACTLY (used only for verification):
        // standard castling letters in trailing-zero order "kqKQ", ep always '-'.
        std::string debug_fen() const
        {
            std::string fen;
            append_placement(fen);
            fen += ' ';
            fen += stm ? 'b' : 'w';
            fen += ' ';
            if (rights == 0)
            {
                fen += '-';
            }
            else
            {
                static const char R[4] = {'k', 'q', 'K', 'Q'};
                std::uint8_t r = rights;
                while (r)
                {
                    int q = mctz64(r);
                    r &= r - 1;
                    fen += R[q];
                }
            }
            fen += " - ";
            fen += std::to_string(halfm);
            fen += ' ';
            fen += std::to_string(fullm);
            return fen;
        }

        // FEN fed to chess::Position::fromFen for actual training. Castling and
        // en-passant are emitted as '-': neither affects NNUE value features, and
        // dropping castling makes the FEN safe for FRC/DFRC positions.
        std::string sf_fen() const
        {
            std::string fen;
            append_placement(fen);
            fen += ' ';
            fen += stm ? 'b' : 'w';
            fen += " - - ";
            fen += std::to_string(halfm);
            fen += ' ';
            fen += std::to_string(fullm);
            return fen;
        }
    };

    // Decode montyformat CompressedChessBoard (4 quad-bitboards) -> 8 bitboards.
    inline void decode_quad_bbs(const std::uint64_t qbbs[4], std::uint64_t bb[8])
    {
        const std::uint64_t blc = qbbs[0];
        const std::uint64_t rqk = qbbs[1];
        const std::uint64_t nbk = qbbs[2];
        const std::uint64_t pbq = qbbs[3];

        const std::uint64_t occ = rqk | nbk | pbq;
        const std::uint64_t pnb = occ ^ qbbs[1];
        const std::uint64_t prq = occ ^ qbbs[2];
        const std::uint64_t nrk = occ ^ qbbs[3];

        bb[0] = occ ^ blc;          // white
        bb[1] = blc;                // black
        bb[2] = pnb & prq;          // pawn
        bb[3] = pnb & nrk;          // knight
        bb[4] = pnb & nbk & pbq;    // bishop
        bb[5] = prq & nrk;          // rook
        bb[6] = pbq & prq & rqk;    // queen
        bb[7] = nbk & rqk;          // king
    }

    // Faithful port of montyformat Castling::from_raw.
    inline void castling_from_raw(const MPosition& pos, std::uint8_t rf[2][2], MCastling& c)
    {
        const bool all_zero = (rf[0][0] | rf[0][1] | rf[1][0] | rf[1][1]) == 0;
        if (all_zero) { c.rook_files[0][0] = 0; c.rook_files[0][1] = 7;
                        c.rook_files[1][0] = 0; c.rook_files[1][1] = 7; }
        else          { c.rook_files[0][0] = rf[0][0]; c.rook_files[0][1] = rf[0][1];
                        c.rook_files[1][0] = rf[1][0]; c.rook_files[1][1] = rf[1][1]; }

        c.chess960 = false; // from_raw never sets it; make() is FRC-correct via rook_files
        for (int i = 0; i < 64; ++i) c.castle_mask[i] = 15;
        c.castle_mask[c.rook_files[0][0]]      = 7;
        c.castle_mask[c.rook_files[0][1]]      = 11;
        c.castle_mask[c.rook_files[1][0] + 56] = 13;
        c.castle_mask[c.rook_files[1][1] + 56] = 14;
        c.castle_mask[pos.king_sq(0)]          = 3;
        c.castle_mask[pos.king_sq(1)]          = 12;
    }

    // win-prob (stm-relative, [0,1]) -> stm-relative centipawns.
    // Same logistic Monty uses in value.rs, with no white-flip (policy score is
    // already stm-relative). Clamped (in float, matching Monty) into int16 range.
    inline std::int16_t score_to_cp(float p)
    {
        if (p < 1e-6f) p = 1e-6f;
        if (p > 1.0f - 1e-6f) p = 1.0f - 1e-6f;
        double pp = static_cast<double>(p);
        double cp = -(400.0 * std::log(1.0 / pp - 1.0));
        cp = std::round(cp);
        if (cp >  32000.0) cp =  32000.0;
        if (cp < -32000.0) cp = -32000.0;
        return static_cast<std::int16_t>(cp);
    }

    // Gini impurity of the MCTS policy (visit distribution): 1 - sum(p_i^2).
    inline double policy_gini(const std::uint8_t* visits, int n)
    {
        long sum = 0;
        for (int i = 0; i < n; ++i) sum += visits[i];
        if (sum <= 0) return 0.0;
        const double inv = 1.0 / static_cast<double>(sum);
        double s2 = 0.0;
        for (int i = 0; i < n; ++i)
        {
            const double p = static_cast<double>(visits[i]) * inv;
            s2 += p * p;
        }
        return 1.0 - s2;
    }

    // Sharp-policy threshold (~median Gini); see read/decode. Plus every capture
    // and in-check position is dropped regardless of Gini. 0 disables the Gini test.
    inline constexpr double MONTY_GINI_MIN = 0.9;

    // best-effort Monty move -> chess::Move (only consumed by optional filters).
    inline chess::Move monty_move_to_sf(std::uint16_t mov, bool stm)
    {
        const int from = mov >> 10;
        const int to   = (mov >> 4) & 63;
        const int flag = mov & 15;
        auto sq = [](int s) {
            return chess::Square(static_cast<chess::File>(s & 7),
                                 static_cast<chess::Rank>(s >> 3));
        };
        const chess::Color color = stm ? chess::Color::Black : chess::Color::White;

        if (flag == 2) return chess::Move::castle(chess::CastleType::Short, color);
        if (flag == 3) return chess::Move::castle(chess::CastleType::Long, color);
        if (flag == 5) return chess::Move::enPassant(sq(from), sq(to));
        if (flag >= 8)
        {
            const chess::PieceType pt =
                static_cast<chess::PieceType>(1 + (flag & 3)); // 1=N,2=B,3=R,4=Q
            return chess::Move::promotion(sq(from), sq(to), chess::Piece(pt, color));
        }
        return chess::Move::normal(sq(from), sq(to));
    }

    // Sniff the 4-byte "BINP" magic that begins every Stockfish .binpack chunk.
    inline bool looks_like_sf_binpack(const std::string& filename)
    {
        std::ifstream f(filename, std::ios::in | std::ios::binary);
        char m[4] = {0, 0, 0, 0};
        f.read(m, 4);
        return f.gcount() == 4 && m[0] == 'B' && m[1] == 'I' && m[2] == 'N' && m[3] == 'P';
    }

    struct MontyFenInputStream : BasicSfenInputStream
    {
        static constexpr auto openmode = std::ios::in | std::ios::binary;
        static constexpr std::size_t QUEUE_CAPACITY = 512;       // chunks buffered
        static constexpr std::size_t CHUNK_SIZE     = 256;       // entries per chunk
        static constexpr int         READ_GAMES_PER_LOCK = 16;   // amortize read lock
        using Chunk = std::vector<TrainingDataEntry>;

        // shuffle_buffer_bytes>0 enables a bullet/Monty-style block shuffle (see
        // the shuffler/slicer below). It is the TOTAL host RAM for the shuffle;
        // it is split across the two ping-pong blocks (so each block holds
        // shuffle_buffer_bytes/2 worth of entries), mirroring bullet/Monty's
        // `buffer_size_mb*MB/sizeof/2`. 0 keeps the original game-order behaviour
        // (used by tools / the verifier, which need deterministic order).
        MontyFenInputStream(std::string filename, bool cyclic,
                            std::function<bool(const TrainingDataEntry&)> skipPredicate,
                            int rank = 0, int world_size = 1, int concurrency = 1,
                            std::size_t shuffle_buffer_bytes = 0) :
            m_filename(std::move(filename)),
            m_stream(m_filename, openmode),
            m_skipPredicate(std::move(skipPredicate)),
            m_cyclic(cyclic),
            m_rank(rank),
            m_world_size(world_size < 1 ? 1 : world_size),
            m_shuffle_enabled(shuffle_buffer_bytes > 0),
            m_rng(0x9E3779B97F4A7C15ull ^
                  (static_cast<std::uint64_t>(rank) * 0xD1B54A32D192ED03ull))
        {
            m_queue.reserve_internal(CHUNK_SIZE);
            const int nprod = concurrency < 1 ? 1 : concurrency;
            if (!m_stream)
            {
                m_active_producers.store(0);
                m_eof.store(true);
                m_output_done.store(true);
                return;
            }

            if (m_shuffle_enabled)
            {
                m_decoded_queue.reserve_internal(CHUNK_SIZE);
                m_block_cap = shuffle_buffer_bytes / 2 / sizeof(TrainingDataEntry);
                if (m_block_cap < CHUNK_SIZE) m_block_cap = CHUNK_SIZE;
                m_block[0].reserve(m_block_cap);
                m_block[1].reserve(m_block_cap);
                m_free_idx.push_back(0);
                m_free_idx.push_back(1);
                m_producer_queue = &m_decoded_queue; // producers -> shuffler
                m_slicer   = std::thread([this]() { slicer_loop(); });
                m_shuffler = std::thread([this]() { shuffler_loop(); });
            }
            else
            {
                m_producer_queue = &m_queue;         // producers -> consumers
            }

            m_active_producers.store(nprod);
            m_producers.reserve(nprod);
            for (int i = 0; i < nprod; ++i)
                m_producers.emplace_back([this]() { producer_loop(); });
        }

        std::optional<TrainingDataEntry> next() override
        {
            std::lock_guard<std::mutex> lk(m_next_mutex);
            for (;;)
            {
                if (m_next_idx < m_next_buf.size())
                    return m_next_buf[m_next_idx++];
                m_next_buf.clear();
                m_next_idx = 0;
                m_next_buf.reserve(CHUNK_SIZE);
                if (!m_queue.take(m_next_buf, [this]() { return queue_done(); }))
                {
                    m_eof.store(true);
                    return std::nullopt;
                }
            }
        }

        void fill(std::vector<TrainingDataEntry>& vec, std::size_t n) override
        {
            while (vec.size() < n)
            {
                Chunk chunk;
                chunk.reserve(CHUNK_SIZE);
                if (!m_queue.take(chunk, [this]() { return queue_done(); }))
                {
                    m_eof.store(true);
                    break;
                }
                for (auto& e : chunk)
                    vec.push_back(std::move(e));
            }
        }

        // Queue is already thread-safe; do NOT take the base fill_lock (it would
        // serialize all consumers and defeat the point).
        void fill_threadsafe(std::vector<TrainingDataEntry>& vec, std::size_t n) override
        {
            fill(vec, n);
        }

        bool eof() const override { return m_eof.load(); }

        ~MontyFenInputStream() override
        {
            m_stop.store(true);
            // Wake everyone blocked on any queue (put or take) or handoff cv.
            m_decoded_queue.signal_stop(true);
            m_queue.signal_stop(true);
            { std::lock_guard<std::mutex> lk(m_hand_mtx); }
            m_free_cv.notify_all();
            m_ready_cv.notify_all();
            for (auto& t : m_producers)
                if (t.joinable())
                    t.join();
            if (m_shuffler.joinable()) m_shuffler.join();
            if (m_slicer.joinable())   m_slicer.join();
        }

    private:
        // Consumers drain m_queue. Without shuffle it is fed by the producers, so
        // it is done once they all exit. With shuffle it is fed by the slicer, so
        // it is done once the slicer signals output_done.
        bool queue_done() const
        {
            if (m_stop.load()) return true;
            if (m_shuffle_enabled) return m_output_done.load();
            return m_active_producers.load() == 0;
        }

        // The shuffler drains m_decoded_queue; it is done once all producers exit.
        bool shuffler_input_done() const { return m_stop.load() || m_producers_done.load(); }
        // The slicer drains m_ready_idx; it is done once the shuffler exits.
        bool slicer_input_done()   const { return m_stop.load() || m_shuffler_done.load(); }

        void reopen_locked()
        {
            m_stream = std::ifstream(m_filename, openmode);
            m_game_counter = 0;
        }

        // Read one game's raw bytes (43-byte header + per-move 5B + visit bytes,
        // ending at the 2-byte NULL terminator) into blob. false on short read.
        bool read_raw_into(std::vector<std::uint8_t>& blob)
        {
            blob.clear();
            std::uint8_t hdr[43];
            m_stream.read(reinterpret_cast<char*>(hdr), 43);
            if (m_stream.gcount() != 43) return false;
            blob.insert(blob.end(), hdr, hdr + 43);
            for (;;)
            {
                std::uint8_t mv[2];
                m_stream.read(reinterpret_cast<char*>(mv), 2);
                if (m_stream.gcount() != 2) return false;
                blob.push_back(mv[0]);
                blob.push_back(mv[1]);
                if (mv[0] == 0 && mv[1] == 0) break; // 2-byte terminator
                std::uint8_t rest[3];
                m_stream.read(reinterpret_cast<char*>(rest), 3);
                if (m_stream.gcount() != 3) return false;
                blob.push_back(rest[0]); blob.push_back(rest[1]); blob.push_back(rest[2]);
                const int nm = rest[2];
                if (nm > 0)
                {
                    const std::size_t s = blob.size();
                    blob.resize(s + nm);
                    m_stream.read(reinterpret_cast<char*>(blob.data()) + s, nm);
                    if (m_stream.gcount() != nm) return false;
                }
            }
            return true;
        }

        // Skip one game's raw bytes (sharding). false on short read.
        bool skip_raw()
        {
            std::uint8_t hdr[43];
            m_stream.read(reinterpret_cast<char*>(hdr), 43);
            if (m_stream.gcount() != 43) return false;
            for (;;)
            {
                std::uint8_t mv[2];
                m_stream.read(reinterpret_cast<char*>(mv), 2);
                if (m_stream.gcount() != 2) return false;
                if (mv[0] == 0 && mv[1] == 0) break;
                std::uint8_t rest[3];
                m_stream.read(reinterpret_cast<char*>(rest), 3);
                if (m_stream.gcount() != 3) return false;
                const int nm = rest[2];
                if (nm > 0) { m_stream.seekg(nm, std::ios::cur); if (!m_stream) return false; }
            }
            return true;
        }

        // Get the next game assigned to this rank into blob. Must hold m_read_mutex.
        // Handles DDP sharding and cyclic reopen. false only at true (non-cyclic) EOF.
        bool next_assigned_game_locked(std::vector<std::uint8_t>& blob)
        {
            for (;;)
            {
                if (m_stop.load()) return false;
                if (m_stream.peek() == std::char_traits<char>::eof() || !m_stream)
                {
                    if (m_cyclic) { reopen_locked(); if (!m_stream) return false; continue; }
                    return false;
                }
                const bool assigned = (m_game_counter % m_world_size) == m_rank;
                ++m_game_counter;
                if (assigned)
                {
                    if (read_raw_into(blob)) return true;
                    if (m_cyclic) { reopen_locked(); continue; } // truncated tail
                    return false;
                }
                if (!skip_raw())
                {
                    if (m_cyclic) { reopen_locked(); continue; }
                    return false;
                }
            }
        }

        // Decode one game's raw bytes into out (filtered + skipPredicate'd). No lock.
        void decode_blob(const std::vector<std::uint8_t>& blob, Chunk& out)
        {
            const std::uint8_t* p   = blob.data();
            const std::uint8_t* end = p + blob.size();
            auto rd = [&](auto& v) -> bool {
                if (p + sizeof(v) > end) return false;
                std::memcpy(&v, p, sizeof(v));
                p += sizeof(v);
                return true;
            };

            std::uint64_t qbbs[4];
            for (int i = 0; i < 4; ++i) if (!rd(qbbs[i])) return;
            std::uint8_t stm_u8, enp_u8, rights_u8, halfm_u8;
            std::uint16_t fullm_u16;
            if (!rd(stm_u8) || !rd(enp_u8) || !rd(rights_u8) || !rd(halfm_u8) || !rd(fullm_u16)) return;

            MPosition pos;
            decode_quad_bbs(qbbs, pos.bb);
            pos.stm = stm_u8 > 0; pos.enp_sq = enp_u8; pos.rights = rights_u8;
            pos.halfm = halfm_u8; pos.fullm = fullm_u16;

            std::uint8_t rf[2][2];
            for (int s = 0; s < 2; ++s) for (int k = 0; k < 2; ++k) if (!rd(rf[s][k])) return;
            MCastling cast;
            castling_from_raw(pos, rf, cast);

            std::uint8_t result_u8;
            if (!rd(result_u8)) return;
            const int white_result = static_cast<int>(result_u8) - 1;

            for (;;)
            {
                std::uint16_t best_move;
                if (!rd(best_move)) break;
                if (best_move == 0) break; // NULL terminator

                std::uint16_t score_raw;
                std::uint8_t num_moves;
                if (!rd(score_raw) || !rd(num_moves)) break;

                double gini = 0.0;
                if (num_moves > 0)
                {
                    if (p + num_moves > end) break;
                    gini = policy_gini(p, num_moves);
                    p += num_moves;
                }

                const bool is_capture = (best_move & 4) != 0;
                if (!(gini < MONTY_GINI_MIN) && !is_capture)
                {
                    chess::Position sfp = chess::Position::fromFen(pos.sf_fen().c_str());
                    if (!sfp.isCheck())
                    {
                        const int stmv = pos.stm ? 1 : 0;
                        TrainingDataEntry e;
                        e.pos    = sfp;
                        e.move   = monty_move_to_sf(best_move, pos.stm);
                        e.score  = score_to_cp(static_cast<float>(score_raw) /
                                               static_cast<float>(0xFFFF));
                        e.ply    = static_cast<std::uint16_t>(2 * (static_cast<int>(pos.fullm) - 1) + stmv);
                        e.result = static_cast<std::int16_t>(stmv ? -white_result : white_result);
                        if (!m_skipPredicate || !m_skipPredicate(e))
                            out.push_back(std::move(e));
                    }
                }

                pos.make(best_move, cast);
            }
        }

        void producer_loop()
        {
            Chunk chunk;
            chunk.reserve(CHUNK_SIZE);
            std::vector<std::vector<std::uint8_t>> blobs;
            blobs.reserve(READ_GAMES_PER_LOCK);
            bool running = true;

            while (running && !m_stop.load())
            {
                blobs.clear();
                bool eof = false;
                {
                    std::lock_guard<std::mutex> lk(m_read_mutex);
                    for (int i = 0; i < READ_GAMES_PER_LOCK && !m_stop.load(); ++i)
                    {
                        std::vector<std::uint8_t> b;
                        if (!next_assigned_game_locked(b)) { eof = true; break; }
                        blobs.push_back(std::move(b));
                    }
                }
                for (auto& b : blobs)
                {
                    decode_blob(b, chunk);
                    if (chunk.size() >= CHUNK_SIZE)
                    {
                        if (!m_producer_queue->put(chunk, [this]() { return m_stop.load(); })) { running = false; break; }
                        chunk.clear();
                    }
                }
                if (eof) running = false;
            }

            if (!chunk.empty() && !m_stop.load())
                m_producer_queue->put(chunk, [this]() { return m_stop.load(); });
            if (m_active_producers.fetch_sub(1) == 1)
            {
                // Last producer: no more decoded data will be produced. Wake the
                // shuffler (shuffle path) or the consumers (no-shuffle path).
                m_producers_done.store(true);
                m_producer_queue->signal_stop(false);
            }
        }

        // ---- shuffle pipeline (only used when m_shuffle_enabled) ---------------
        //
        // Producers feed decoded entries into m_decoded_queue (game order). The
        // single shuffler thread draws them and builds a large block using the
        // inside-out Fisher-Yates so the block is uniformly shuffled as it fills
        // (no separate shuffle pass -> never stalls the decoders). Full blocks are
        // handed, via a 2-slot ping-pong (m_free_idx / m_ready_idx), to the slicer
        // thread, which chops them into chunks for the consumers. Two blocks are
        // live at once (one filling, one draining) -> total RAM is the configured
        // shuffle_buffer_bytes, matching bullet/Monty.

        // Block-index handoff. Returns false only when shutting down.
        bool take_free(int& out)
        {
            std::unique_lock<std::mutex> lk(m_hand_mtx);
            m_free_cv.wait(lk, [this]() { return !m_free_idx.empty() || m_stop.load(); });
            if (m_free_idx.empty()) return false;
            out = m_free_idx.front();
            m_free_idx.pop_front();
            return true;
        }
        void put_free(int idx)
        {
            { std::lock_guard<std::mutex> lk(m_hand_mtx); m_free_idx.push_back(idx); }
            m_free_cv.notify_one();
        }
        void put_ready(int idx)
        {
            { std::lock_guard<std::mutex> lk(m_hand_mtx); m_ready_idx.push_back(idx); }
            m_ready_cv.notify_one();
        }
        // Returns false when no more full blocks will arrive (shuffler done / stop).
        bool take_ready(int& out)
        {
            std::unique_lock<std::mutex> lk(m_hand_mtx);
            m_ready_cv.wait(lk, [this]() { return !m_ready_idx.empty() || slicer_input_done(); });
            if (m_ready_idx.empty()) return false;
            out = m_ready_idx.front();
            m_ready_idx.pop_front();
            return true;
        }

        void shuffler_loop()
        {
            int cur = -1;
            if (!take_free(cur)) { finish_shuffler(); return; }
            m_block[cur].clear();

            Chunk in;
            in.reserve(CHUNK_SIZE);
            bool more = true;
            while (more && !m_stop.load())
            {
                in.clear();
                if (!m_decoded_queue.take(in, [this]() { return shuffler_input_done(); })) { more = false; break; }
                for (auto& e : in)
                {
                    std::vector<TrainingDataEntry>& buf = m_block[cur];
                    const std::size_t i = buf.size();
                    const std::size_t j = static_cast<std::size_t>(m_rng() % (i + 1));
                    if (j == i) buf.push_back(std::move(e));            // inside-out:
                    else { buf.push_back(std::move(buf[j]));            //   move slot j to end,
                           buf[j] = std::move(e); }                     //   put new entry at j
                    if (buf.size() >= m_block_cap)
                    {
                        put_ready(cur);
                        cur = -1;
                        if (!take_free(cur)) { more = false; break; }
                        m_block[cur].clear();
                    }
                }
            }
            // Non-cyclic EOF: emit the trailing partial block (still uniformly
            // shuffled thanks to the inside-out invariant). Cyclic training never
            // reaches here.
            if (!m_stop.load() && cur >= 0 && !m_block[cur].empty())
                put_ready(cur);
            finish_shuffler();
        }

        void finish_shuffler()
        {
            m_shuffler_done.store(true);
            m_ready_cv.notify_all(); // no more ready blocks; wake slicer
        }

        void slicer_loop()
        {
            Chunk c;
            c.reserve(CHUNK_SIZE);
            int idx = -1;
            bool running = true;
            while (running && !m_stop.load())
            {
                if (!take_ready(idx)) break;
                std::vector<TrainingDataEntry>& buf = m_block[idx];
                for (std::size_t off = 0; off < buf.size() && !m_stop.load(); off += CHUNK_SIZE)
                {
                    const std::size_t end = std::min(off + CHUNK_SIZE, buf.size());
                    c.clear();
                    for (std::size_t k = off; k < end; ++k) c.push_back(std::move(buf[k]));
                    if (!m_queue.put(c, [this]() { return m_stop.load(); })) { running = false; break; }
                }
                buf.clear();
                if (running)
                    put_free(idx);
            }
            m_output_done.store(true);
            m_queue.signal_stop(false); // no more output; wake consumers
        }

        std::string   m_filename;
        std::ifstream m_stream;
        std::function<bool(const TrainingDataEntry&)> m_skipPredicate;
        bool m_cyclic;
        int  m_rank;
        int  m_world_size;

        std::mutex m_read_mutex;
        long long  m_game_counter = 0; // guarded by m_read_mutex

        thread_safe_types::ThreadSafeRingBuffer<Chunk, QUEUE_CAPACITY> m_queue; // -> consumers
        std::vector<std::thread> m_producers;
        std::atomic<int>  m_active_producers{0};
        std::atomic<bool> m_producers_done{false};
        std::atomic<bool> m_stop{false};
        std::atomic<bool> m_eof{false};

        // Where producers push decoded chunks: &m_queue (no shuffle) or
        // &m_decoded_queue (shuffle). Set once in the constructor.
        thread_safe_types::ThreadSafeRingBuffer<Chunk, QUEUE_CAPACITY>* m_producer_queue = nullptr;

        // ---- shuffle pipeline state (only used when m_shuffle_enabled) --------
        bool        m_shuffle_enabled = false;
        std::size_t m_block_cap = 0;                    // entries per ping-pong block
        thread_safe_types::ThreadSafeRingBuffer<Chunk, QUEUE_CAPACITY> m_decoded_queue; // producers -> shuffler
        std::vector<TrainingDataEntry> m_block[2];      // the two ping-pong blocks
        // Block-index handoff between shuffler and slicer (only ever holds 0/1).
        std::mutex              m_hand_mtx;
        std::condition_variable m_free_cv;   // shuffler waits here for a free block
        std::condition_variable m_ready_cv;  // slicer waits here for a full block
        std::deque<int>         m_free_idx;  // blocks the shuffler may fill
        std::deque<int>         m_ready_idx; // full blocks for the slicer
        std::thread m_shuffler;
        std::thread m_slicer;
        std::atomic<bool> m_shuffler_done{false};
        std::atomic<bool> m_output_done{false};
        std::mt19937_64   m_rng;                        // used only by the shuffler thread

        std::mutex  m_next_mutex; // serializes next() (non-parallel path)
        Chunk       m_next_buf;
        std::size_t m_next_idx = 0;
    };

} // namespace monty
} // namespace training_data

#endif
