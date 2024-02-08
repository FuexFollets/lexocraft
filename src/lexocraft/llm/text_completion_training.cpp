#include <Eigen/Eigen>

#include <lexocraft/llm/lexer.hpp>
#include <lexocraft/llm/text_completion.hpp>
#include <lexocraft/llm/text_completion_training.hpp>

namespace lc {
    /*
    TextCompletionTrainer::TrainingModification TextCompletionTrainer::train_neural_network(
        const std::vector<std::string>& training_data_sections,
        const std::optional<CostWeightCoefficients>& cost_weight_coefficients) {
        assert(text_completer.has_value() && "Text completer not set");
    }
    */

    float TextCompletionTrainer::calculate_prediction_costs(
        const std::string& training_data_section,
        const std::optional<CostWeightCoefficients>& cost_weight_coefficients) {
        assert(text_completer.has_value() && "Text completer not set");

        const std::vector<grammar::Token> tokens = text_completer->tokenize(training_data_section);

        const float section_sentence_length_mean = sentence_length_mean(tokens);
        const float section_sentence_length_stddev = sentence_length_stddev(tokens);
        const float section_flesch_kincaid_level =
            TextCompleter::flesch_kincaid_level(training_data_section);
        const float section_sentence_count = sentence_count(tokens);

        float cost_sum {};

        for (std::size_t token_index {}; token_index < tokens.size(); ++token_index) {
            const grammar::Token& token = tokens [token_index];

            lc::TextCompleter::EphemeralMemoryNNOutput output {
                text_completer->predict_next_token_value(
                    token, section_sentence_length_mean, section_sentence_length_stddev,
                    section_flesch_kincaid_level, section_sentence_count)};

            const bool is_last_token = token_index == tokens.size() - 1;
            const float is_last_token_float_quantity = is_last_token ? 1.0F : 0.0F;
            const float predicted_is_end = output.is_end;

            const grammar::Token::Type token_type = token.type;
            const Eigen::VectorXf token_type_float_quantities = {
                token_type == grammar::Token::Type::Homogeneous ? 1.0F : 0.0F,
                token_type == grammar::Token::Type::Alphanumeric ? 1.0F : 0.0F,
                token_type == grammar::Token::Type::Digit ? 1.0F : 0.0F,
                token_type == grammar::Token::Type::Symbol ? 1.0F : 0.0F};
            const Eigen::VectorXf predicted_token_type_values = {
                output.token_is_homogeneous, output.token_is_alphanumeric, output.token_is_digit,
                output.token_is_symbol};

            const TextCompleter::SearchedWordVector searched_word_vector =
                std::get<0>(text_completer->find_word_vector(token.value));

            const Eigen::VectorXf token_word_vector_value = searched_word_vector.word_vector.vector;
            const Eigen::VectorXf predicted_word_vector = output.word_vector_value;

            const float section_termination_cost = is_last_token_float_quantity - predicted_is_end;

            const float token_type_cost =
                (token_type_float_quantities - predicted_token_type_values).norm();

            const float word_vector_cost = (token_word_vector_value - predicted_word_vector).norm();

            const CostWeightCoefficients cost_weight_coefficients_value =
                cost_weight_coefficients.value_or(DEFAULT_COST_WEIGHT_COEFFICIENTS);

            const float cost =
                cost_weight_coefficients_value.incorrect_token_type * token_type_cost +
                cost_weight_coefficients_value.incorrect_section_termination *
                    section_termination_cost +
                cost_weight_coefficients_value.predicted_word_vector_euclidean_distance_magnitude *
                    word_vector_cost;

            cost_sum += cost;
        }

        return cost_sum / tokens.size();
    }

    float TextCompletionTrainer::calculate_prediction_costs(
        const std::vector<std::string>& training_data_sections,
        const std::optional<CostWeightCoefficients>& cost_weight_coefficients) {
        float cost_sum {};

        for (const std::string& training_data_section: training_data_sections) {
            cost_sum += calculate_prediction_costs(training_data_section, cost_weight_coefficients);
        }

        return cost_sum / training_data_sections.size();
    }
} // namespace lc
