#include <Eigen/Eigen>

#include <lexocraft/llm/text_completion.hpp>
#include <lexocraft/llm/vector_database.hpp>

namespace lc {
    WordVector TextCompleter::improvised_word_vector(
        const std::string& word,
        const std::vector<VectorDatabase::SearchResult>& word_vectors_search_result) {
        Eigen::VectorXf word_vector_value;

        for (const VectorDatabase::SearchResult& result: word_vectors_search_result) {
            WordVectorImproviserNNFields fields {result, ephemeral_memory, word_vector_value};

            WordVectorImproviserNNOutput output;

            assert(output.from_output(fields.to_vector()));
        }

        return WordVector {word, word_vector_value};
    }
} // namespace lc
