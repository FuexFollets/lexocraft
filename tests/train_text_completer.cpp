#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <icecream.hpp>
#include <nanobench.h>

#include <lexocraft/llm/text_completion.hpp>
#include <lexocraft/llm/text_completion_training.hpp>
#include <lexocraft/neural_network/neural_network.hpp>

int main(const int argc, const char** argv) {
    std::vector<std::string> args {std::next(argv, 1), std::next(argv, argc)};

    std::cout << "args: " << args.size() << "\n";

    for (std::size_t index = 0; index < args.size(); ++index) {
        std::cout << "arg[" << index << "]: " << args [index] << "\n";
    }

    const std::string text_completion_filepath = args.at(0);
    const std::string training_data_filepath = args.at(1);

    lc::TextCompleter text_completer;

    std::cout << "Loading text completion file at: " << text_completion_filepath << "\n";
    text_completer.load_file(text_completion_filepath);

    IC();
    IC(text_completer.word_vector_improviser_fields_sizes.word_vector_search_result);

    std::cout << "Constructing text completion trainer\n";
    lc::TextCompletionTrainer text_completion_trainer {text_completer};

    std::cout << "Loading training data file at: " << training_data_filepath << "\n";
    const std::ifstream training_data_file {training_data_filepath};
    std::stringstream training_data_stream;
    training_data_stream << training_data_file.rdbuf();
    const std::string training_data = training_data_stream.str();

    std::cout << "Training neural network\n";
    const auto result = text_completion_trainer.train_neural_network(training_data, 1);

    IC();
    IC(result.improved_cost);
    IC(result.original_cost);

    if (result.ephemeral_memory_diff.has_value()) {
        std::cout << "ephemeral_memory_diff: " << result.ephemeral_memory_diff.value() << "\n";
    }

    else {
        std::cout << "ephemeral_memory_diff: no value\n";
    }

    if (result.context_builder_diff.has_value()) {
        std::cout << "context_builder_diff: " << result.context_builder_diff.value() << "\n";
    }

    else {
        std::cout << "context_builder_diff: no value\n";
    }

    if (result.word_vector_improviser_diff.has_value()) {
        std::cout << "word_vector_improviser_diff: " << result.word_vector_improviser_diff.value()
                  << "\n";
    }

    else {
        std::cout << "word_vector_improviser_diff: no value\n";
    }

    return 0;
}
