#include <algorithm>
#include <fstream>
#include <functional>
#include <optional>
#include <vector>

#include <cereal/archives/binary.hpp>
#include <cereal/cereal.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/vector.hpp>
#include <rapidfuzz/fuzz.hpp>

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

    std::vector<int> WordVector::soundex() const {
        std::vector<int> soundex;

        for (char letter: word) {
            if (SOUNDEX_CODES.contains(letter)) {
                soundex.push_back(SOUNDEX_CODES.at(letter));
            }
        }

        return soundex;
    }

    float WordVector::levenshtein_distance(const WordVector& other) const {
        return lc::levenshtein_distance(word, other.word);
    }

    float WordVector::soundex_distance(const WordVector& other) const {
        return lc::levenshtein_distance(soundex(), other.soundex());
    }

    float WordVector::similarity(const WordVector& other, float soundex_weight,
                                 float levenshtein_weight) const {
        const float soundex_distance = this->soundex_distance(other);
        const float levenshtein_distance = this->levenshtein_distance(other);
        const float weight_sum = soundex_weight + levenshtein_weight;

        return 1 - (soundex_weight * soundex_distance + levenshtein_weight * levenshtein_distance) /
                       weight_sum;
    }

    VectorDatabase::VectorDatabase(const std::vector<WordVector>& words) : words(words) {
        for (const WordVector& word: words) {
            word_map [word.word] = word;
        }
    }

    void VectorDatabase::add_word(const std::string& word, bool randomize_vector) {
        // (WordVector {std::string {word}, randomize_vector})

        WordVector new_word {std::string {word}, randomize_vector};
        words.push_back(new_word);
        word_map [word] = new_word;
    }

    void VectorDatabase::save(const std::filesystem::path& filepath) const {
        std::ofstream file {filepath};

        cereal::BinaryOutputArchive oarchive {file};

        oarchive(*this);
    }

    void VectorDatabase::load(const std::filesystem::path& filepath) {
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

    [[deprecated(
        "VectorDatabase::rapidfuzz_search_closest_n")]] std::vector<VectorDatabase::SearchResult>
        VectorDatabase::search_closest_n(const std::string& searched_word, int top_n,
                                         float threshold, float soundex_weight,
                                         float levenshtein_weight,
                                         bool stop_when_top_n_are_found) const {
        std::vector<SearchResult> results;

        results.reserve(top_n);

        float lowest_similarity_in_top_n = 0.0F;

        const std::function<bool()> results_are_full = [&]() {
            return results.size() == static_cast<std::size_t>(top_n);
        };

        const WordVector searched_word_ {searched_word};

        for (const WordVector& word: words) {
            const float similarity =
                searched_word_.similarity(word, soundex_weight, levenshtein_weight);

            if (similarity >= threshold) {
                lowest_similarity_in_top_n = similarity;
            }

            const bool was_added =
                add_search_result(results, {word, similarity}, top_n, lowest_similarity_in_top_n);

            if (was_added && results_are_full() && stop_when_top_n_are_found) {
                break;
            }
        }

        std::sort(results.begin(), results.end(),
                  [](const SearchResult& first_result, const SearchResult& second_result) {
                      return first_result.similarity < second_result.similarity;
                  });

        return results;
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
