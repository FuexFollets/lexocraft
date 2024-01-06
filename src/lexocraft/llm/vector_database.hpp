#ifndef LEXOCRAFT_VECTOR_DATABASE_HPP
#define LEXOCRAFT_VECTOR_DATABASE_HPP

#include <filesystem>
#include <string>
#include <vector>

#include <Eigen/Eigen>

#include <lexocraft/cereal_eigen.hpp>

namespace lc {
    class WordVector {
        public:

        static constexpr std::size_t WORD_VECTOR_DIMENSIONS = 32;

        std::string word;
        Eigen::Vector<float, WORD_VECTOR_DIMENSIONS> vector;

        template <class Archive>
        void serialize(Archive& archive) {
            archive(word, vector);
        }
    };

    class VectorDatabase {
        public:

        VectorDatabase() = default;
        explicit VectorDatabase(const std::vector<WordVector>& words);
        explicit VectorDatabase(const std::filesystem::path& filepath);

        std::vector<WordVector> words;

        // void save(const std::filesystem::path& filepath) const;
        void load(const std::filesystem::path& filepath);

        template <class Archive>
        void serialize(Archive& archive) {
            archive(words);
        }
    };
} // namespace lc

#endif // LEXOCRAFT_VECTOR_DATABASE_HPP
