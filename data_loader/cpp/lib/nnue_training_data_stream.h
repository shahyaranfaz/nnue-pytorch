#ifndef _SFEN_STREAM_H_
#define _SFEN_STREAM_H_

#include "parallel_dataloader.h"

#include <optional>
#include <bit>
#include <cstdint>
#include <fstream>
#include <string>
#include <memory>
#include <stdexcept>
#include <vector>
#include <functional>
#include <atomic>
#include <mutex>

namespace training_data {

    using namespace binpack;

    static bool ends_with(const std::string& lhs, const std::string& end)
    {
        if (end.size() > lhs.size()) return false;

        return std::equal(end.rbegin(), end.rend(), lhs.rbegin());
    }

    static bool has_extension(const std::string& filename, const std::string& extension)
    {
        return ends_with(filename, "." + extension);
    }

    struct BasicSfenInputStream
    {
        virtual std::optional<TrainingDataEntry> next() = 0;
        virtual void fill(std::vector<TrainingDataEntry>& vec, std::size_t n)
        {
            for (std::size_t i = 0; i < n; ++i)
            {
                auto v = this->next();
                if (!v.has_value())
                {
                    break;
                }
                vec.emplace_back(*v);
            }
        }
        virtual void fill_threadsafe(std::vector<TrainingDataEntry>& vec, std::size_t n)
        {
            std::lock_guard<std::mutex> lock(fill_lock);
            this->fill(vec, n);
        }

        virtual bool eof() const = 0;
        virtual ~BasicSfenInputStream() {}

    private:
        std::mutex fill_lock;
    };

    struct BinSfenInputStream : BasicSfenInputStream
    {
        static constexpr auto openmode = std::ios::in | std::ios::binary;
        static inline const std::string extension = "bin";

        BinSfenInputStream(std::string filename, bool cyclic, std::function<bool(const TrainingDataEntry&)> skipPredicate) :
            m_stream(filename, openmode),
            m_filename(filename),
            m_eof(!m_stream),
            m_cyclic(cyclic),
            m_skipPredicate(std::move(skipPredicate))
        {
        }

        std::optional<TrainingDataEntry> next() override
        {
            nodchip::PackedSfenValue e;
            bool reopenedFileOnce = false;
            for(;;)
            {
                if(m_stream.read(reinterpret_cast<char*>(&e), sizeof(nodchip::PackedSfenValue)))
                {
                    auto entry = packedSfenValueToTrainingDataEntry(e);
                    if (!m_skipPredicate || !m_skipPredicate(entry))
                        return entry;
                }
                else
                {
                    if (m_cyclic)
                    {
                        if (reopenedFileOnce)
                            return std::nullopt;

                        m_stream = std::fstream(m_filename, openmode);
                        reopenedFileOnce = true;
                        if (!m_stream)
                            return std::nullopt;

                        continue;
                    }

                    m_eof.store(true, std::memory_order_release);
                    return std::nullopt;
                }
            }
        }

        bool eof() const override
        {
            return m_eof.load();
        }

        ~BinSfenInputStream() override {}

    private:
        std::fstream m_stream;
        std::string m_filename;
        std::atomic<bool> m_eof;
        bool m_cyclic;
        std::function<bool(const TrainingDataEntry&)> m_skipPredicate;
    };

    #pragma pack(push, 1)
    struct BulletFormatEntry
    {
        std::uint64_t occupancy;
        std::uint8_t pieces[16];
        std::int16_t score;
        std::uint8_t result;
        std::uint8_t king_square;
        std::uint8_t opp_king_square;
        std::uint8_t padding[3];
    };
    #pragma pack(pop)

    static_assert(sizeof(BulletFormatEntry) == 32);

    inline TrainingDataEntry bulletFormatToTrainingDataEntry(
        const BulletFormatEntry& packed
    )
    {
        if (packed.result > 2)
            throw std::runtime_error("invalid Bullet result");
        if (std::popcount(packed.occupancy) > 32)
            throw std::runtime_error("too many pieces in Bullet record");

        chess::Board board;
        std::uint64_t occupancy = packed.occupancy;
        int piece_index = 0;
        while (occupancy)
        {
            const int square = std::countr_zero(occupancy);
            occupancy &= occupancy - 1;

            const std::uint8_t byte = packed.pieces[piece_index / 2];
            const std::uint8_t code = piece_index % 2 == 0
                                    ? byte & 0x0F
                                    : byte >> 4;
            const int piece_type = code & 0x07;
            if (piece_type > static_cast<int>(chess::PieceType::King))
                throw std::runtime_error("invalid Bullet piece code");

            const auto color = (code & 0x08) != 0
                             ? chess::Color::Black
                             : chess::Color::White;
            board.place(
                chess::Piece(static_cast<chess::PieceType>(piece_type), color),
                chess::Square(square)
            );
            ++piece_index;
        }

        if (!board.isValid())
            throw std::runtime_error("invalid board in Bullet record");
        if (static_cast<int>(board.kingSquare(chess::Color::White))
            != packed.king_square)
            throw std::runtime_error("white king mismatch in Bullet record");

        TrainingDataEntry entry{};
        entry.pos = chess::Position(
            board,
            chess::Color::White,
            chess::Square::none(),
            chess::CastlingRights::None
        );
        entry.score = packed.score;
        entry.ply = 0;
        entry.result = static_cast<std::int16_t>(packed.result) - 1;
        return entry;
    }

    struct BulletSfenInputStream : BasicSfenInputStream
    {
        static constexpr auto openmode = std::ios::in | std::ios::binary;
        static inline const std::string extension = "bullet";

        BulletSfenInputStream(
            std::string filename,
            bool cyclic,
            std::function<bool(const TrainingDataEntry&)> skipPredicate
        ) :
            m_stream(filename, openmode),
            m_filename(std::move(filename)),
            m_eof(!m_stream),
            m_cyclic(cyclic),
            m_skipPredicate(std::move(skipPredicate))
        {
            if (!m_stream)
                return;
            m_stream.seekg(0, std::ios::end);
            const auto size = m_stream.tellg();
            if (size < 0 || size % sizeof(BulletFormatEntry) != 0)
                throw std::runtime_error(
                    "Bullet file size is not a multiple of 32 bytes"
                );
            m_stream.seekg(0, std::ios::beg);
        }

        std::optional<TrainingDataEntry> next() override
        {
            BulletFormatEntry packed{};
            bool reopened_file_once = false;
            for (;;)
            {
                if (m_stream.read(
                    reinterpret_cast<char*>(&packed), sizeof(packed)
                ))
                {
                    auto entry = bulletFormatToTrainingDataEntry(packed);
                    if (!m_skipPredicate || !m_skipPredicate(entry))
                        return entry;
                }
                else
                {
                    if (m_cyclic)
                    {
                        if (reopened_file_once)
                            return std::nullopt;
                        m_stream = std::fstream(m_filename, openmode);
                        reopened_file_once = true;
                        if (!m_stream)
                            return std::nullopt;
                        continue;
                    }

                    m_eof.store(true, std::memory_order_release);
                    return std::nullopt;
                }
            }
        }

        bool eof() const override { return m_eof.load(); }

    private:
        std::fstream m_stream;
        std::string m_filename;
        std::atomic<bool> m_eof;
        bool m_cyclic;
        std::function<bool(const TrainingDataEntry&)> m_skipPredicate;
    };

    struct BinpackSfenInputStream : BasicSfenInputStream
    {
        static constexpr auto openmode = std::ios::in | std::ios::binary;
        static inline const std::string extension = "binpack";

        BinpackSfenInputStream(std::string filename, bool cyclic, std::function<bool(const TrainingDataEntry&)> skipPredicate) :
            m_stream(std::make_unique<binpack::CompressedTrainingDataEntryReader>(filename, openmode)),
            m_filename(filename),
            m_eof(!m_stream->hasNext()),
            m_cyclic(cyclic),
            m_skipPredicate(std::move(skipPredicate))
        {
        }

        std::optional<TrainingDataEntry> next() override
        {
            bool reopenedFileOnce = false;
            for(;;)
            {
                if (!m_stream->hasNext())
                {
                    if (m_cyclic)
                    {
                        if (reopenedFileOnce)
                            return std::nullopt;

                        m_stream = std::make_unique<binpack::CompressedTrainingDataEntryReader>(m_filename, openmode);
                        reopenedFileOnce = true;

                        if (!m_stream->hasNext())
                            return std::nullopt;

                        continue;
                    }

                    m_eof.store(true, std::memory_order_release);
                    return std::nullopt;
                }

                auto e = m_stream->next();
                if (!m_skipPredicate || !m_skipPredicate(e))
                    return e;
            }
        }

        bool eof() const override
        {
            return m_eof.load();
        }

        ~BinpackSfenInputStream() override {}

    private:
        std::unique_ptr<binpack::CompressedTrainingDataEntryReader> m_stream;
        std::string m_filename;
        std::atomic<bool> m_eof;
        bool m_cyclic;
        std::function<bool(const TrainingDataEntry&)> m_skipPredicate;
    };

    struct BinpackSfenInputParallelStream : BasicSfenInputStream
    {
        static constexpr auto openmode = std::ios::in | std::ios::binary;
        static inline const std::string extension = "binpack";

        BinpackSfenInputParallelStream(int concurrency, const std::vector<std::string>& filenames, bool cyclic, std::function<bool(const TrainingDataEntry&)> skipPredicate, int rank = 0, int world_size = 1) :
            m_stream(std::make_unique<binpack::CompressedTrainingDataEntryParallelReader>(concurrency, filenames, openmode, cyclic, skipPredicate, rank, world_size)),
            m_filenames(filenames),
            m_eof(false),
            m_concurrency(concurrency),
            m_cyclic(cyclic),
            m_skipPredicate(skipPredicate)
        {
        }

        std::optional<TrainingDataEntry> next() override
        {
            // filtering is done a layer deeper.
            auto v = m_stream->next();
            if (!v.has_value())
            {
                m_eof.store(true, std::memory_order_release);
                return std::nullopt;
            }

            return v;
        }

        void fill(std::vector<TrainingDataEntry>& v, std::size_t n) override
        {
            fill_threadsafe(v, n);
        }

        void fill_threadsafe(std::vector<TrainingDataEntry>& v, std::size_t n) override
        {
            std::size_t k = static_cast<size_t>(m_stream->fill(v, n));
            if (n != k)
            {
                m_eof.store(true, std::memory_order_release);
            }
        }

        bool eof() const override
        {
            return m_eof.load();
        }

        ~BinpackSfenInputParallelStream() override {}

    private:
        std::unique_ptr<binpack::CompressedTrainingDataEntryParallelReader> m_stream;
        std::vector<std::string> m_filenames;
        std::atomic<bool> m_eof;
        int m_concurrency;
        bool m_cyclic;
        std::function<bool(const TrainingDataEntry&)> m_skipPredicate;
    };

    inline std::unique_ptr<BasicSfenInputStream> open_sfen_input_file(const std::string& filename, bool cyclic, std::function<bool(const TrainingDataEntry&)> skipPredicate = nullptr)
    {
        if (has_extension(filename, BinSfenInputStream::extension))
            return std::make_unique<BinSfenInputStream>(filename, cyclic, std::move(skipPredicate));
        else if (has_extension(filename, BulletSfenInputStream::extension))
            return std::make_unique<BulletSfenInputStream>(filename, cyclic, std::move(skipPredicate));
        else if (has_extension(filename, BinpackSfenInputStream::extension))
            return std::make_unique<BinpackSfenInputStream>(filename, cyclic, std::move(skipPredicate));

        return nullptr;
    }

    inline std::unique_ptr<BasicSfenInputStream> open_sfen_input_file_parallel(int concurrency, const std::vector<std::string>& filenames, bool cyclic, std::function<bool(const TrainingDataEntry&)> skipPredicate = nullptr, int rank = 0, int world_size = 1)
    {
        // TODO (low priority): optimize and parallelize .bin reading.
        if (has_extension(filenames[0], BinSfenInputStream::extension))
            return std::make_unique<BinSfenInputStream>(filenames[0], cyclic, std::move(skipPredicate));
        else if (has_extension(filenames[0], BulletSfenInputStream::extension))
        {
            if (filenames.size() != 1)
                throw std::runtime_error(
                    "Bullet input currently supports exactly one file"
                );
            return std::make_unique<BulletSfenInputStream>(
                filenames[0], cyclic, std::move(skipPredicate)
            );
        }
        else if (has_extension(filenames[0], BinpackSfenInputParallelStream::extension))
            return std::make_unique<BinpackSfenInputParallelStream>(concurrency, filenames, cyclic, std::move(skipPredicate), rank, world_size);

        return nullptr;
    }
}

#endif
