#include <Eigen/Eigen>

#include <lexocraft/fancy_eigen_print.hpp>
#include <lexocraft/llm/text_completion.hpp>
#include <lexocraft/llm/vector_database.hpp>

namespace lc {
    WordVector TextCompleter::improvised_word_vector(
        const std::string& word,
        const std::vector<VectorDatabase::SearchResult>& word_vectors_search_result) {
        Eigen::VectorXf word_vector_value =
            Eigen::VectorXf::Zero(word_vector_improviser_fields_sizes.word_vector_value);
        ephemeral_memory =
            Eigen::VectorXf::Zero(word_vector_improviser_fields_sizes.ephemeral_memory);

        for (const VectorDatabase::SearchResult& result: word_vectors_search_result) {
            WordVectorImproviserNNFields fields(result, ephemeral_memory, word_vector_value,
                                                word_vector_improviser_fields_sizes);

            WordVectorImproviserNNOutput output(word_vector_improviser_output_sizes);

            assert(output.from_output(word_vector_improviser.compute(fields.to_vector())));
            word_vector_value = output.word_vector_value;
        }

        return WordVector {word, word_vector_value};
    }

    TextCompleter& TextCompleter::reset_ephemeral_memory() {
        ephemeral_memory.setZero();

        return *this;
    }

    Eigen::VectorXf TextCompleter::accumulate_context_memory(float sentence_length_mean,
                                                             float sentence_length_stddev,
                                                             float flesch_kincaid_grade) {
        ContextBuilderNNFields fields(sentence_length_mean, sentence_length_stddev,
                                      flesch_kincaid_grade, ephemeral_memory, context_memory,
                                      context_builder_fields_sizes);

        ContextBuilderNNOutput output(context_builder_output_sizes);

        const auto compute_result {context_builder.compute(fields.to_vector())};

        assert(output.from_output(compute_result));

        this->reset_ephemeral_memory();
        this->context_memory = output.context_memory;

        return output.context_memory;
    }

    TextCompleter::EphemeralMemoryNNOutput TextCompleter::predict_next_token_value(
        const grammar::Token& token, float sentence_length_mean_, float sentence_length_stddev_,
        float flesch_kincaid_grade_, float sentence_count_) {
        const SearchedWordVector word_vector_result = find_word_vector(token.value);

        const EphemeralMemoryNNFields fields(
            sentence_length_mean_, sentence_length_stddev_, flesch_kincaid_grade_, sentence_count_,
            word_vector_result, ephemeral_memory, context_memory, ephemeral_memory_fields_sizes);

        EphemeralMemoryNNOutput output(ephemeral_memory_output_sizes);

        assert(output.from_output(ephemeral_memory_accmulator.compute(fields.to_vector())));

        this->ephemeral_memory = output.ephemeral_memory;

        return output;
    }
} // namespace lc
