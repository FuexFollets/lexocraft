#ifndef LEXOCRAFT_VECTOR_DATABASE_HPP
#define LEXOCRAFT_VECTOR_DATABASE_HPP

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include <cereal/types/string.hpp>
#include <cereal/types/vector.hpp>
#include <Eigen/Eigen>
#include <mapbox/eternal.hpp>
#include <tsl/robin_map.h>
#include <annoy/annoylib.h>

#include <lexocraft/cereal_eigen.hpp>

namespace lc {
    class WordVector {
        public:

        static constexpr std::size_t WORD_VECTOR_DIMENSIONS = 32;

        using Vector_t = Eigen::Vector<float, WORD_VECTOR_DIMENSIONS>;

        WordVector() = default;
        WordVector(const WordVector&) = default;
        WordVector(WordVector&&) = default;
        WordVector& operator=(const WordVector&) = default;
        WordVector& operator=(WordVector&&) = default;

        WordVector(std::string&& word, Vector_t&& vector);
        WordVector(const std::string& word, const Vector_t& vector);
        explicit WordVector(std::string&& word, bool randomize_vector = true);
        explicit WordVector(const std::string& word, bool randomize_vector = true);

        std::string word;
        Eigen::Vector<float, WORD_VECTOR_DIMENSIONS> vector;

        [[nodiscard]] float
            similarity(const WordVector& other) const; /* Magnitude squared (0.0 - 1.0) */

        [[nodiscard]] float
            similarity(const Eigen::VectorXf& other) const; /* Magnitude squared (0.0 - 1.0) */

        template <class Archive>
        void serialize(Archive& archive) {
            archive(word, vector);
        }
    };

    class VectorDatabase {
        public:

        /*
    using RobinMap_t =
        tsl::robin_map<std::string, WordVector, std::hash<std::string>, std::equal_to<>,
                       std::allocator<std::pair<std::string, int>>, true>;
                       */

        using RobinMap_t = tsl::robin_map<std::string, WordVector>;

        // all default constructors
        VectorDatabase() = default;
        VectorDatabase(const VectorDatabase&) = default;
        VectorDatabase(VectorDatabase&&) = default;
        VectorDatabase& operator=(const VectorDatabase&) = default;
        VectorDatabase& operator=(VectorDatabase&&) = default;

        explicit VectorDatabase(const std::vector<WordVector>& words);

        std::vector<WordVector> words;
        RobinMap_t word_map {};

        void add_word(const std::string& word, bool randomize_vector = true);
        void add_word(const WordVector& word, bool replace_existing = true);

        void save(const std::filesystem::path& filepath) const;
        void load(const std::filesystem::path& filepath);

        struct SearchResult {
            WordVector word;
            float similarity;
        };

        [[nodiscard]] std::vector<SearchResult>
            rapidfuzz_search_closest_n(const std::string& searched_word, int top_n,
                                       float threshold = 0.9F,
                                       bool stop_when_top_n_are_found = true) const;

        [[nodiscard]] std::vector<SearchResult>
            search_closest_vector_value_n(const WordVector& searched_vector, int top_n,
                                          float threshold, bool stop_when_top_n_are_found) const;

        [[nodiscard]] std::vector<SearchResult>
            search_closest_vector_value_n(const Eigen::VectorXf& searched_vector, int top_n,
                                          float threshold, bool stop_when_top_n_are_found) const;

        [[nodiscard]] std::optional<WordVector> search_from_map(const std::string& word) const;

        [[nodiscard]] std::size_t longest_element() const;

        template <class Archive>
        void serialize(Archive& archive) {
            archive(words);
        }
    };

    bool add_search_result(std::vector<VectorDatabase::SearchResult>& results,
                           const VectorDatabase::SearchResult& word, int max_result_count,
                           std::optional<float> maybe_least_relevant_search_result);
} // namespace lc

#endif // LEXOCRAFT_VECTOR_DATABASE_HPP
