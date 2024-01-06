#ifndef LEXOCRAFT_VECTOR_DATABASE_HPP
#define LEXOCRAFT_VECTOR_DATABASE_HPP

#include <filesystem>
#include <string>
#include <vector>

#include <cereal/types/string.hpp>
#include <cereal/types/vector.hpp>
#include <Eigen/Eigen>

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
        explicit WordVector(std::string&& word, bool randomize_vector = true);

        std::string word;
        Eigen::Vector<float, WORD_VECTOR_DIMENSIONS> vector;

        template <class Archive>
        void serialize(Archive& archive) {
            archive(word, vector);
        }
    };

    class VectorDatabase {
        public:

        // all default constructors
        VectorDatabase() = default;
        VectorDatabase(const VectorDatabase&) = default;
        VectorDatabase(VectorDatabase&&) = default;
        VectorDatabase& operator=(const VectorDatabase&) = default;
        VectorDatabase& operator=(VectorDatabase&&) = default;

        explicit VectorDatabase(const std::vector<WordVector>& words);

        std::vector<WordVector> words;

        void add_word(const std::string& word, bool randomize_vector = true);

        void save(const std::filesystem::path& filepath) const;
        void load(const std::filesystem::path& filepath);

        template <class Archive>
        void serialize(Archive& archive) {
            archive(words);
        }
    };
} // namespace lc

#endif // LEXOCRAFT_VECTOR_DATABASE_HPP
