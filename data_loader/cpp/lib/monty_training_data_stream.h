#ifndef _MONTY_STREAM_H_
#define _MONTY_STREAM_H_

// Reader for Monty's "montyformat" *policy* data files (e.g. produced by
// monty/crates/datagen with --policy-data, then interleaved).
//
// Monty stores, per game, a start position + a list of (best_move, win-prob
// score, visit distribution).  The Stockfish NNUE trainer is a *value* trainer,
// so this reader extracts, for every position, exactly what binpack::
// TrainingDataEntry needs and DROPS the policy visit distribution:
//
//   pos    : reconstructed by replaying best moves from the start position
//   score  : win-prob (side-to-move relative, [0,1]) -> stm centipawns
//   result : Monty stores game result white-relative; flipped to stm-relative
//   ply    : 2*(fullm-1) + stm
//   move   : best_move (best-effort; only used by optional "filtered" skipping)
//
// Conventions are taken verbatim from monty/crates/montyformat
// (format.rs, value.rs, chess/{position,moves,frc,consts}.rs) and verified
// position-by-position against Monty's own decoder.  The on-disk format is
// little-endian and uncompressed; this reader assumes a little-endian host
// (x86-64 / aarch64), like the rest of the trainer.

#include "training_data_entry.h" // binpack::TrainingDataEntry + chess::Position (post-#489 split of nnue_training_data_formats.h)

#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
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

        // FEN fed to chess::Position::fromFen for actual training.  Castling and
        // en-passant are emitted as '-': neither affects NNUE value features, and
        // dropping castling makes the FEN safe for FRC/DFRC positions (no need to
        // match rook letters to squares).
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
    // already stm-relative).  Clamped away from 0/1 and into int16 range.
    inline std::int16_t score_to_cp(float p)
    {
        // Clamp in float (matches Monty's f32 math) before widening, so the
        // saturation boundary is identical to the reference decoder.
        if (p < 1e-6f) p = 1e-6f;
        if (p > 1.0f - 1e-6f) p = 1.0f - 1e-6f;
        double pp = static_cast<double>(p);
        double cp = -(400.0 * std::log(1.0 / pp - 1.0));
        cp = std::round(cp);
        if (cp >  32000.0) cp =  32000.0;
        if (cp < -32000.0) cp = -32000.0;
        return static_cast<std::int16_t>(cp);
    }

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

        MontyFenInputStream(std::string filename, bool cyclic,
                            std::function<bool(const TrainingDataEntry&)> skipPredicate,
                            int rank = 0, int world_size = 1) :
            m_stream(filename, openmode),
            m_filename(std::move(filename)),
            m_eof(!m_stream),
            m_cyclic(cyclic),
            m_skipPredicate(std::move(skipPredicate)),
            m_rank(rank),
            m_world_size(world_size < 1 ? 1 : world_size)
        {
        }

        std::optional<TrainingDataEntry> next() override
        {
            std::lock_guard<std::mutex> lk(m_mutex);
            return next_impl();
        }

        void fill(std::vector<TrainingDataEntry>& vec, std::size_t n) override
        {
            std::lock_guard<std::mutex> lk(m_mutex);
            for (std::size_t i = 0; i < n; ++i)
            {
                auto v = next_impl();
                if (!v.has_value()) break;
                vec.emplace_back(*v);
            }
        }

        void fill_threadsafe(std::vector<TrainingDataEntry>& vec, std::size_t n) override
        {
            fill(vec, n);
        }

        bool eof() const override { return m_eof.load(); }

        ~MontyFenInputStream() override {}

    private:
        template <class T> bool read_le(T& v)
        {
            m_stream.read(reinterpret_cast<char*>(&v), sizeof(T));
            return m_stream.gcount() == static_cast<std::streamsize>(sizeof(T));
        }

        std::optional<TrainingDataEntry> next_impl()
        {
            for (;;)
            {
                while (m_buf_idx < m_buf.size())
                {
                    const TrainingDataEntry& e = m_buf[m_buf_idx++];
                    if (!m_skipPredicate || !m_skipPredicate(e))
                        return e;
                }

                m_buf.clear();
                m_buf_idx = 0;

                if (!load_next_assigned_game())
                {
                    if (m_cyclic && !m_reopened_once)
                    {
                        m_stream = std::ifstream(m_filename, openmode);
                        m_reopened_once = true;
                        m_game_counter = 0;
                        if (!m_stream) { m_eof.store(true); return std::nullopt; }
                        continue;
                    }
                    m_eof.store(true);
                    return std::nullopt;
                }
            }
        }

        // Skip exactly one game's bytes, consuming the same span as read_game:
        // a 43-byte header, then per move a 2-byte move + 2-byte score + 1-byte
        // count + count visit bytes, ending at the 2-byte NULL terminator.
        // NB: the terminator is a 2-byte u16 (matches montyformat deserialise_from);
        // it must NOT be read as a 5-byte move header, or framing drifts.
        bool skip_game()
        {
            char header[43];
            m_stream.read(header, 43);
            if (m_stream.gcount() != 43) return false;
            for (;;)
            {
                std::uint16_t best_move;
                if (!read_le(best_move)) return false;
                if (best_move == 0) break;
                std::uint16_t score_raw;
                std::uint8_t num_moves;
                if (!read_le(score_raw) || !read_le(num_moves)) return false;
                if (num_moves > 0) m_stream.seekg(num_moves, std::ios::cur);
            }
            return true;
        }

        // Read & decode one game into m_buf (positions). Returns false at EOF.
        bool read_game()
        {
            std::uint64_t qbbs[4];
            for (int i = 0; i < 4; ++i) if (!read_le(qbbs[i])) return false;

            std::uint8_t  stm_u8, enp_u8, rights_u8, halfm_u8;
            std::uint16_t fullm_u16;
            if (!read_le(stm_u8) || !read_le(enp_u8) || !read_le(rights_u8) ||
                !read_le(halfm_u8) || !read_le(fullm_u16))
                return false;

            MPosition pos;
            decode_quad_bbs(qbbs, pos.bb);
            pos.stm    = stm_u8 > 0;
            pos.enp_sq = enp_u8;
            pos.rights = rights_u8;
            pos.halfm  = halfm_u8;
            pos.fullm  = fullm_u16;

            std::uint8_t rf[2][2];
            for (int s = 0; s < 2; ++s)
                for (int k = 0; k < 2; ++k)
                    if (!read_le(rf[s][k])) return false;

            MCastling cast;
            castling_from_raw(pos, rf, cast);

            std::uint8_t result_u8;
            if (!read_le(result_u8)) return false;
            const int white_result = static_cast<int>(result_u8) - 1; // 0->-1,1->0,2->+1

            for (;;)
            {
                std::uint16_t best_move;
                if (!read_le(best_move)) return false;
                if (best_move == 0) break; // Move::NULL terminates the game

                std::uint16_t score_raw;
                if (!read_le(score_raw)) return false;

                std::uint8_t num_moves;
                if (!read_le(num_moves)) return false;
                if (num_moves > 0) m_stream.seekg(num_moves, std::ios::cur); // drop policy

                const int stmv = pos.stm ? 1 : 0;

                TrainingDataEntry e;
                e.pos    = chess::Position::fromFen(pos.sf_fen().c_str());
                e.move   = monty_move_to_sf(best_move, pos.stm);
                e.score  = score_to_cp(static_cast<float>(score_raw) /
                                       static_cast<float>(0xFFFF));
                e.ply    = static_cast<std::uint16_t>(2 * (static_cast<int>(pos.fullm) - 1) + stmv);
                e.result = static_cast<std::int16_t>(stmv ? -white_result : white_result);
                m_buf.emplace_back(e);

                pos.make(best_move, cast);
            }
            return true;
        }

        // Honour DDP sharding: only games with (index % world_size == rank) are
        // decoded; the rest are skipped.
        bool load_next_assigned_game()
        {
            for (;;)
            {
                m_stream.peek();
                if (!m_stream || m_stream.eof()) return false;

                const bool assigned = (m_game_counter % m_world_size) == m_rank;
                ++m_game_counter;

                if (assigned)
                    return read_game();
                if (!skip_game())
                    return false;
            }
        }

        std::ifstream m_stream;
        std::string   m_filename;
        std::atomic<bool> m_eof;
        bool m_cyclic;
        std::function<bool(const TrainingDataEntry&)> m_skipPredicate;
        int  m_rank;
        int  m_world_size;

        std::mutex m_mutex;
        std::vector<TrainingDataEntry> m_buf;
        std::size_t m_buf_idx = 0;
        long long   m_game_counter = 0;
        bool        m_reopened_once = false;
    };

} // namespace monty
} // namespace training_data

#endif
