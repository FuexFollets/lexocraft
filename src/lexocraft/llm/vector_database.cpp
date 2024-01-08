#include <algorithm>
#include <fstream>
#include <functional>
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

        return (soundex_weight * soundex_distance + levenshtein_weight * levenshtein_distance) /
               weight_sum;
    }

    VectorDatabase::VectorDatabase(const std::vector<WordVector>& words) : words(words) {
    }

    void VectorDatabase::add_word(const std::string& word, bool randomize_vector) {
        words.push_back(WordVector {std::string {word}, randomize_vector});
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
    }

    std::vector<VectorDatabase::SearchResult> VectorDatabase::search_closest_n(
        const WordVector& searched_word, int top_n, float threshold, float soundex_weight,
        float levenshtein_weight, bool stop_when_top_n_are_found) const {
        std::vector<SearchResult> results;

        results.reserve(top_n);

        float lowest_similarity_in_top_n = 1.0F;

        const std::function<bool()> results_are_full = [&]() {
            return results.size() == static_cast<std::size_t>(top_n);
        };

        for (const WordVector& word: words) {
            const float similarity =
                searched_word.similarity(word, soundex_weight, levenshtein_weight);

            if ((similarity < threshold) &&
                (!results_are_full || (similarity < lowest_similarity_in_top_n))) {
                results.emplace_back(word, similarity);
                lowest_similarity_in_top_n = std::min(lowest_similarity_in_top_n, similarity);

                if (results_are_full() && stop_when_top_n_are_found) {
                    break;
                }
            }
        }

        std::sort(results.begin(), results.end(),
                  [](const SearchResult& first_result, const SearchResult& second_result) {
                      return first_result.similarity < second_result.similarity;
                  });

        return results;
    }

    std::vector<VectorDatabase::SearchResult>
        VectorDatabase::rapidfuzz_search_closest_n(const WordVector& searched_word, int top_n,
                                                   float threshold,
                                                   bool stop_when_top_n_are_found) const {
        std::vector<SearchResult> results;

        results.reserve(top_n);

        float lowest_similarity_in_top_n = 1.0F;

        const std::function<bool()> results_are_full = [&]() {
            return results.size() == static_cast<std::size_t>(top_n);
        };

        for (const WordVector& word: words) {
            const float similarity = rapidfuzz::fuzz::ratio(searched_word.word, word.word) / 100.0F;

            if ((similarity < threshold) &&
                (!results_are_full || (similarity < lowest_similarity_in_top_n))) {
                results.emplace_back(word, similarity);
                lowest_similarity_in_top_n = std::min(lowest_similarity_in_top_n, similarity);

                if (results_are_full() && stop_when_top_n_are_found) {
                    break;
                }
            }
        }

        std::sort(results.begin(), results.end(),
                  [](const SearchResult& first_result, const SearchResult& second_result) {
                      return first_result.similarity < second_result.similarity;
                  });

        return results;
    }
} // namespace lc
