#include <BS_thread_pool.hpp>
#include <Eigen/Core>
#include <icecream.hpp>
#include <nanobench.h>

#include <lexocraft/llm/lexer.hpp>
#include <lexocraft/llm/text_completion.hpp>
#include <lexocraft/llm/text_completion_training.hpp>

namespace lc {
    float TextCompletionTrainer::calculate_prediction_costs(
        const std::shared_ptr<TextCompleter>& text_completer,
        const std::string& training_data_section,
        const std::optional<CostWeightCoefficients>& cost_weight_coefficients) {
        const std::vector<grammar::Token> tokens = text_completer->tokenize(training_data_section);

        const float section_sentence_length_mean = sentence_length_mean(tokens);
        const float section_sentence_length_stddev = sentence_length_stddev(tokens);
        const float section_flesch_kincaid_level =
            TextCompleter::flesch_kincaid_level(training_data_section);
        const float section_sentence_count = sentence_count(tokens);

        float cost_sum {};

        IC();
        for (std::size_t token_index {}; token_index < tokens.size(); ++token_index) {
            const grammar::Token& token = tokens [token_index];

            std::optional<lc::TextCompleter::EphemeralMemoryNNOutput> output_option {};

            /*
            lc::TextCompleter::EphemeralMemoryNNOutput output {
                text_completer->predict_next_token_value(
                    token, section_sentence_length_mean, section_sentence_length_stddev,
                    section_flesch_kincaid_level, section_sentence_count)};
                    */

            ankerl::nanobench::Bench().run(
                "text_completer->predict_next_token_value(" + token.value + ")", [&] {
                    output_option = text_completer->predict_next_token_value(
                        token, section_sentence_length_mean, section_sentence_length_stddev,
                        section_flesch_kincaid_level, section_sentence_count);
                });

            assert(output_option.has_value() && "Output option has no value");
            lc::TextCompleter::EphemeralMemoryNNOutput output {output_option.value()};

            const bool is_last_token = token_index == tokens.size() - 1;
            const float is_last_token_float_quantity = is_last_token ? 1.0F : 0.0F;
            const float predicted_is_end = output.is_end;

            const grammar::Token::Type token_type = token.type;
            const Eigen::Vector4f token_type_float_quantities {
                {token_type == grammar::Token::Type::Homogeneous ? 1.0F : 0.0F,
                 token_type == grammar::Token::Type::Alphanumeric ? 1.0F : 0.0F,
                 token_type == grammar::Token::Type::Digit ? 1.0F : 0.0F,
                 token_type == grammar::Token::Type::Symbol ? 1.0F : 0.0F}
            };
            const Eigen::Vector4f predicted_token_type_values = {
                output.token_is_homogeneous, output.token_is_alphanumeric, output.token_is_digit,
                output.token_is_symbol};

            const TextCompleter::SearchedWordVector searched_word_vector =
                std::get<0>(text_completer->find_word_vector(token.value));

            const Eigen::VectorXf token_word_vector_value = searched_word_vector.word_vector.vector;
            const Eigen::VectorXf predicted_word_vector = output.word_vector_value;

            const float section_termination_cost = is_last_token_float_quantity - predicted_is_end;

            const float token_type_cost =
                (token_type_float_quantities - predicted_token_type_values).prod();

            const float word_vector_cost = (token_word_vector_value - predicted_word_vector).prod();

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
        const std::shared_ptr<TextCompleter>& text_completer,
        const std::vector<std::string>& training_data_sections,
        const std::optional<CostWeightCoefficients>& cost_weight_coefficients) {
        float cost_sum {};

        for (const std::string& training_data_section: training_data_sections) {
            cost_sum += TextCompletionTrainer::calculate_prediction_costs(
                text_completer, training_data_section, cost_weight_coefficients);
        }

        return cost_sum / training_data_sections.size();
    }

    TextCompletionTrainer::TrainingModification TextCompletionTrainer::train_neural_network(
        const std::string& training_data, std::size_t threads_count,
        const std::optional<CostWeightCoefficients>& cost_weight_coefficients) {
        assert(text_completer.has_value() && "Text completer not set");

        enum class TrainingModificationType : std::uint8_t {
            EphemeralMemory = 0b001,
            ContextBuilder = 0b010,
            WordVectorImproviser = 0b100
        };

        const std::vector<std::uint8_t> training_modification_combonations {1, 2, 3, 4, 5, 6, 7};
        std::vector<TrainingModification> training_modifications_per_combination(
            training_modification_combonations.size());

        float diff_scale_factor = 0.02F;

        TrainingModification all_training_modifications {
            .ephemeral_memory_diff =
                text_completer->ephemeral_memory_accmulator.random_diff() * diff_scale_factor,
            .context_builder_diff =
                text_completer->context_builder.random_diff() * diff_scale_factor,
            .word_vector_improviser_diff =
                text_completer->word_vector_improviser.random_diff() * diff_scale_factor};

        std::cout << "Creating thread pool with " << threads_count << " threads\n";
        BS::thread_pool thread_pool(threads_count);

        std::cout << "Calculating default cost\n";
        const float default_cost =
            calculate_prediction_costs(std::make_shared<TextCompleter>(text_completer.value()),
                                       training_data, cost_weight_coefficients);

        IC();
        IC(default_cost);
        std::exit(0);

        for (std::size_t index {}; index < training_modification_combonations.size(); ++index) {
            static_cast<void>(thread_pool.submit_task([this,
                                                       &training_modifications_per_combination,
                                                       index, &training_modification_combonations,
                                                       &training_data, &cost_weight_coefficients,
                                                       &all_training_modifications, default_cost] {
                const std::uint8_t training_modification_combonations_value =
                    training_modification_combonations [index];
                TrainingModification training_modification {};

                if ((training_modification_combonations_value &
                     static_cast<std::uint8_t>(TrainingModificationType::EphemeralMemory)) != 0) {
                    training_modification.ephemeral_memory_diff =
                        all_training_modifications.ephemeral_memory_diff;
                }

                if ((training_modification_combonations_value &
                     static_cast<std::uint8_t>(TrainingModificationType::ContextBuilder)) != 0) {
                    training_modification.context_builder_diff =
                        all_training_modifications.context_builder_diff;
                }

                if ((training_modification_combonations_value &
                     static_cast<std::uint8_t>(TrainingModificationType::WordVectorImproviser)) !=
                    0) {
                    training_modification.word_vector_improviser_diff =
                        all_training_modifications.word_vector_improviser_diff;
                }

                IC();
                std::cout << "Testing neural network in thread " << index << "\n";
                TextCompleter modified_text_completer = text_completer.value();
                modified_text_completer =
                    apply_training_modification(modified_text_completer, training_modification);

                const float modified_cost = calculate_prediction_costs(
                    std::make_shared<TextCompleter>(modified_text_completer), training_data,
                    cost_weight_coefficients);

                training_modification.original_cost = default_cost;
                training_modification.improved_cost = modified_cost;

                training_modifications_per_combination [index] = training_modification;
            }));
        }

        thread_pool.wait();

        float lowest_cost {default_cost};
        std::optional<std::size_t> lowest_cost_index {};

        for (std::size_t index {}; index < training_modification_combonations.size(); ++index) {
            if (training_modifications_per_combination [index].improved_cost < lowest_cost) {
                lowest_cost = training_modifications_per_combination [index].improved_cost;
                lowest_cost_index = index;
            }
        }

        if (lowest_cost_index.has_value()) {
            return training_modifications_per_combination [lowest_cost_index.value()];
        }

        return {};
    }

    TextCompletionTrainer::TrainingModification TextCompletionTrainer::train_neural_network(
        const std::vector<std::string>& training_data_sections, std::size_t threads_count,
        const std::optional<CostWeightCoefficients>& cost_weight_coefficients) {
        assert(text_completer.has_value() && "Text completer not set");

        enum class TrainingModificationType : std::uint8_t {
            EphemeralMemory = 0b001,
            ContextBuilder = 0b010,
            WordVectorImproviser = 0b100
        };

        const std::vector<std::uint8_t> training_modification_combonations {0, 1, 2, 3, 4, 5, 6, 7};

        std::vector<TrainingModification> training_modifications_per_combination(
            training_modification_combonations.size());

        float diff_scale_factor = 0.02F;

        TrainingModification all_training_modifications {
            .ephemeral_memory_diff =
                text_completer->ephemeral_memory_accmulator.random_diff() * diff_scale_factor,
            .context_builder_diff =
                text_completer->context_builder.random_diff() * diff_scale_factor,
            .word_vector_improviser_diff =
                text_completer->word_vector_improviser.random_diff() * diff_scale_factor};

        std::vector<float> costs(training_modification_combonations.size());

        TextCompleter::VectorDatabasePointerCollection_t database_collection_pointers {
            text_completer->get_vector_database_pointers()};

        text_completer->vector_database = nullptr;
        text_completer->create_vector_subdatabases();

        BS::thread_pool thread_pool(threads_count);

        for (std::size_t index {}; index < training_modification_combonations.size(); ++index) {
            static_cast<void>(thread_pool.submit_task(
                [this, &training_modifications_per_combination, &costs, index,
                 &training_modification_combonations, &training_data_sections,
                 &cost_weight_coefficients, &all_training_modifications,
                 &database_collection_pointers] {
                    const std::uint8_t training_modification_combonations_value =
                        training_modification_combonations [index];
                    TrainingModification training_modification {};

                    if ((training_modification_combonations_value &
                         static_cast<std::uint8_t>(TrainingModificationType::EphemeralMemory)) !=
                        0) {
                        training_modification.ephemeral_memory_diff =
                            all_training_modifications.ephemeral_memory_diff;
                    }

                    if ((training_modification_combonations_value &
                         static_cast<std::uint8_t>(TrainingModificationType::ContextBuilder)) !=
                        0) {
                        training_modification.context_builder_diff =
                            all_training_modifications.context_builder_diff;
                    }

                    if ((training_modification_combonations_value &
                         static_cast<std::uint8_t>(
                             TrainingModificationType::WordVectorImproviser)) != 0) {
                        training_modification.word_vector_improviser_diff =
                            all_training_modifications.word_vector_improviser_diff;
                    }

                    TextCompleter modified_text_completer = text_completer.value();
                    modified_text_completer =
                        apply_training_modification(modified_text_completer, training_modification);
                    modified_text_completer.assign_vector_database_pointers(
                        database_collection_pointers);

                    costs [index] = calculate_prediction_costs(
                        std::make_shared<TextCompleter>(modified_text_completer),
                        training_data_sections, cost_weight_coefficients);

                    training_modifications_per_combination [index] = training_modification;
                }));
        }

        thread_pool.wait();

        std::size_t lowest_cost_index {};

        for (std::size_t index {}; index < costs.size(); ++index) {
            if (costs [index] < costs [lowest_cost_index]) {
                lowest_cost_index = index;
            }
        }

        return training_modifications_per_combination [lowest_cost_index];
    }

    TextCompleter& TextCompletionTrainer::apply_training_modification(
        TextCompleter& text_completer, const TrainingModification& training_modification) {
        if (training_modification.context_builder_diff.has_value()) {
            text_completer.context_builder.modify(
                training_modification.context_builder_diff.value());
        }

        if (training_modification.ephemeral_memory_diff.has_value()) {
            text_completer.ephemeral_memory_accmulator.modify(
                training_modification.ephemeral_memory_diff.value());
        }

        if (training_modification.word_vector_improviser_diff.has_value()) {
            text_completer.word_vector_improviser.modify(
                training_modification.word_vector_improviser_diff.value());
        }

        return text_completer;
    }
} // namespace lc
