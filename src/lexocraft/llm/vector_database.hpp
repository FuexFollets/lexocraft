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

#include <lexocraft/cereal_eigen.hpp>

namespace lc {
    class WordVector {
        public:

        static constexpr std::size_t WORD_VECTOR_DIMENSIONS = 32;

        using Vector_t = Eigen::Vector<float, WORD_VECTOR_DIMENSIONS>;

        [[deprecated]] static MAPBOX_ETERNAL_CONSTEXPR const auto SOUNDEX_CODES =
            mapbox::eternal::map<char, int>({
  // clang-format off
                {'b', 1}, {'f', 1}, {'p', 1}, {'v', 1},
                {'c', 2}, {'g', 2}, {'j', 2}, {'k', 2}, {'q', 2}, {'s', 2}, {'x', 2},
                {'d', 3}, {'t', 3},
                {'l', 4},
                {'m', 5}, {'n', 5},
                {'r', 6},
  // clang-format on
        });

        WordVector() = default;
        WordVector(const WordVector&) = default;
        WordVector(WordVector&&) = default;
        WordVector& operator=(const WordVector&) = default;
        WordVector& operator=(WordVector&&) = default;

        WordVector(std::string&& word, Vector_t&& vector);
        explicit WordVector(std::string&& word, bool randomize_vector = true);
        explicit WordVector(const std::string& word, bool randomize_vector = true);

        std::string word;
        Eigen::Vector<float, WORD_VECTOR_DIMENSIONS> vector;

        [[nodiscard]] std::vector<int> soundex() const;
        [[deprecated]] [[nodiscard]] float soundex_distance(const WordVector& other) const;
        [[deprecated]] [[nodiscard]] float levenshtein_distance(const WordVector& other) const;
        [[deprecated]] [[nodiscard]] float similarity(const WordVector& other,
                                                      float soundex_weight = 0.5F,
                                                      float levenshtein_weight = 0.5F) const;

        template <class Archive>
        void serialize(Archive& archive) {
            archive(word, vector);
        }
    };

    class VectorDatabase {
        public:

        using RobinMap_t =
            tsl::robin_map<std::string, WordVector, std::hash<std::string>, std::equal_to<>,
                           std::allocator<std::pair<std::string, int>>, true>;

        // all default constructors
        VectorDatabase() = default;
        VectorDatabase(const VectorDatabase&) = default;
        VectorDatabase(VectorDatabase&&) = default;
        VectorDatabase& operator=(const VectorDatabase&) = default;
        VectorDatabase& operator=(VectorDatabase&&) = default;

        explicit VectorDatabase(const std::vector<WordVector>& words);

        std::vector<WordVector> words;
        RobinMap_t word_map;

        void add_word(const std::string& word, bool randomize_vector = true);

        void save(const std::filesystem::path& filepath) const;
        void load(const std::filesystem::path& filepath);

        struct SearchResult {
            WordVector word;
            float similarity;
        };

        [[nodiscard]] std::vector<SearchResult>
            search_closest_n(const std::string& searched_word, int top_n, float threshold = 0.2F,
                             float soundex_weight = 0.5F, float levenshtein_weight = 0.5F,
                             bool stop_when_top_n_are_found = true) const;

        [[nodiscard]] std::vector<SearchResult>
            rapidfuzz_search_closest_n(const std::string& searched_word, int top_n,
                                       float threshold = 0.2F,
                                       bool stop_when_top_n_are_found = true) const;

        [[nodiscard]] std::optional<WordVector> search_from_map(const std::string& word) const;

        template <class Archive>
        void serialize(Archive& archive) {
            archive(words);
        }
    };

    bool add_search_result(std::vector<VectorDatabase::SearchResult>& results,
                           const VectorDatabase::SearchResult& word, int max_result_count,
                           std::optional<float> maybe_least_relevant_search_result);

    template <typename FirstContainer, typename SecondContainer>
    float levenshtein_distance(const FirstContainer& first, const SecondContainer& second) {
        // Implemented just like the comment

        using ValueType = typename FirstContainer::value_type;

        const int first_length = first.size();
        const int second_length = second.size();

        if (first_length == 0 || second_length == 0) {
            return std::max(first_length, second_length);
        }

        std::vector<std::vector<ValueType>> distance_matrix(
            first_length + 1, std::vector<ValueType>(second_length + 1));

        for (int index = 0; index <= first_length; index++) {
            distance_matrix [index][0] = index;
        }

        for (int index = 0; index <= second_length; index++) {
            distance_matrix [0][index] = index;
        }

        for (int index1 = 1; index1 <= first_length; index1++) {
            for (int index2 = 1; index2 <= second_length; index2++) {
                ValueType cost = (first [index1 - 1] == second [index2 - 1]) ? 0 : 1;

                distance_matrix [index1][index2] =
                    std::min({distance_matrix [index1 - 1][index2] + 1,
                              distance_matrix [index1][index2 - 1] + 1,
                              distance_matrix [index1 - 1][index2 - 1] + cost});
            }
        }

        const int distance = distance_matrix [first_length][second_length];
        const int max_possible_distance = first_length + second_length;

        return static_cast<float>(distance) / max_possible_distance;
    }
} // namespace lc

#endif // LEXOCRAFT_VECTOR_DATABASE_HPP
