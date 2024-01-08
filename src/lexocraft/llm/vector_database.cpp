#include <algorithm>
#include <fstream>
#include <vector>

#include <cereal/archives/binary.hpp>
#include <cereal/cereal.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/vector.hpp>

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

    std::vector<float> WordVector::soundex() const {
        std::vector<float> soundex;

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
} // namespace lc
