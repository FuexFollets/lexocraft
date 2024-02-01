#include <fstream>
#include <functional>
#include <optional>
#include <vector>

#include <cereal/archives/binary.hpp>
#include <cereal/cereal.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/vector.hpp>
#include <rapidfuzz/fuzz.hpp>
#include <tsl/robin_map.h>

#include <lexocraft/cereal_eigen.hpp>
#include <lexocraft/llm/vector_database.hpp>

namespace lc {
    WordVector::WordVector(std::string&& word, Vector_t&& vector) :
        word(std::move(word)), vector(std::move(vector)) {
    }

    WordVector::WordVector(const std::string& word, const Vector_t& vector) :
        word(word), vector(vector) {
    }

    WordVector::WordVector(std::string&& word, bool randomize_vector) : word(std::move(word)) {
        if (randomize_vector) {
            vector = WordVector::Vector_t::Random();
        }

        else {
            vector = WordVector::Vector_t::Zero();
        }
    }

    WordVector::WordVector(const std::string& word, bool randomize_vector) : word(word) {
        if (randomize_vector) {
            vector = WordVector::Vector_t::Random();
        }

        else {
            vector = WordVector::Vector_t::Zero();
        }
    }

    float WordVector::similarity(const WordVector& other) const {
        return similarity(other.vector);
    }

    float WordVector::similarity(const Eigen::VectorXf& other) const {
        assert(other.size() == WORD_VECTOR_DIMENSIONS);

        const Vector_t difference = vector - other;

        return 1 - (difference.squaredNorm() / (WORD_VECTOR_DIMENSIONS * 4));
    }

    const float* WordVector::data() const {
        return vector.data();
    }

    VectorDatabase::VectorDatabase(const std::vector<WordVector>& words) : words(words) {
        for (const WordVector& word: words) {
            word_map [word.word] = word;
        }
    }

    void VectorDatabase::add_word(const std::string& word, bool randomize_vector) {
        // (WordVector {std::string {word}, randomize_vector})

        const WordVector new_word {std::string {word}, randomize_vector};
        words.push_back(new_word);
        word_map [word] = new_word;
        const int index = words.size() - 1;
        annoy_index->add_item(index, new_word.vector.data());
    }

    void VectorDatabase::add_word(const WordVector& word, bool replace_existing) {
        const bool already_has_word = word_map.contains(word.word);

        if (replace_existing && already_has_word) {
            word_map [word.word] = word;
        }

        else if (!already_has_word) {
            words.push_back(word);
            word_map [word.word] = word;
        }
    }

    void VectorDatabase::save_file(const std::filesystem::path& filepath) const {
        std::ofstream file {filepath};

        cereal::BinaryOutputArchive oarchive {file};

        oarchive(*this);
    }

    void VectorDatabase::load_file(const std::filesystem::path& filepath) {
        std::ifstream file {filepath};

        cereal::BinaryInputArchive iarchive {file};

        iarchive(*this);

        // Add words to map

        for (const WordVector& word: words) {
            word_map [word.word] = word;
        }
    }

    bool add_search_result(std::vector<VectorDatabase::SearchResult>& results,
                           const VectorDatabase::SearchResult& word, int max_result_count,
                           std::optional<float> maybe_least_relevant_search_result) {
        bool result_is_inserted {false};

        if (const std::optional<float> least_relevant_search_result =
                maybe_least_relevant_search_result) {
            const float least_relevant_similarity = least_relevant_search_result.value();

            if (results.size() >= static_cast<std::size_t>(max_result_count) &&
                word.similarity < least_relevant_similarity) {
                return false;
            }
        }

        for (std::size_t index {0}; index < results.size(); ++index) {
            if (word.similarity > results.at(index).similarity) {
                results.insert(std::next(results.begin(), index), word);
                result_is_inserted = true;

                break;
            }
        }

        if (!result_is_inserted && results.size() < static_cast<std::size_t>(max_result_count)) {
            results.push_back(word);

            return true;
        }

        if (result_is_inserted && results.size() > static_cast<std::size_t>(max_result_count)) {
            results.erase(results.begin() + max_result_count, results.end());

            return true;
        }

        return false;
    }

    std::vector<VectorDatabase::SearchResult>
        VectorDatabase::rapidfuzz_search_closest_n(const std::string& searched_word, int top_n,
                                                   float threshold,
                                                   bool stop_when_top_n_are_found) const {
        std::vector<SearchResult> results;

        results.reserve(top_n);

        float lowest_similarity_in_top_n = 0.0F;

        const std::function<bool()> results_are_full = [&]() {
            return results.size() == static_cast<std::size_t>(top_n);
        };

        for (const WordVector& word: words) {
            const float similarity = rapidfuzz::fuzz::ratio(searched_word, word.word) / 100.0F;

            if (similarity < threshold) {
                continue;
            }

            const bool was_added =
                add_search_result(results, {word, similarity}, top_n, lowest_similarity_in_top_n);

            if (was_added && results_are_full() && stop_when_top_n_are_found) {
                break;
            }
        }

        return results;
    }

    std::vector<VectorDatabase::SearchResult>
        VectorDatabase::search_closest_vector_value_n(const WordVector& searched_vector, int top_n,
                                                      float threshold,
                                                      bool stop_when_top_n_are_found) const {
        return search_closest_vector_value_n(searched_vector.vector, top_n, threshold,
                                             stop_when_top_n_are_found);
    }

    std::vector<VectorDatabase::SearchResult>
        VectorDatabase::search_closest_vector_value_n(const Eigen::VectorXf& searched_vector,
                                                      int top_n, float threshold,
                                                      bool stop_when_top_n_are_found) const {
        std::vector<SearchResult> results;

        results.reserve(top_n);

        float lowest_similarity_in_top_n = 0.0F;

        const std::function<bool()> results_are_full = [&]() {
            return results.size() == static_cast<std::size_t>(top_n);
        };

        for (const WordVector& word: words) {
            const float similarity = word.similarity(searched_vector);

            if (similarity < threshold) {
                continue;
            }

            const bool was_added =
                add_search_result(results, {word, similarity}, top_n, lowest_similarity_in_top_n);

            if (was_added && results_are_full() && stop_when_top_n_are_found) {
                break;
            }
        }

        return results;
    }

    std::optional<WordVector> VectorDatabase::search_from_map(const std::string& word) const {
        if (!word_map.contains(word)) {
            return std::nullopt;
        }

        return word_map.at(word);
    }

    std::size_t VectorDatabase::longest_element() const {
        return std::max_element(words.begin(), words.end(),
                                [](const WordVector& first_word, const WordVector& second_word) {
                                    return first_word.word.size() < second_word.word.size();
                                })
            ->word.size();
    }
} // namespace lc
