#include <ios>
#include <iostream>

#include <lexocraft/fancy_eigen_print.hpp>
#include <lexocraft/llm/text_completion.hpp>
#include <lexocraft/llm/vector_database.hpp>

int main(int argc, char** argv) {
    std::vector<std::string> args {std::next(argv, 1), std::next(argv, argc)};
    std::cout << std::boolalpha;

    std::cout << "args: " << args.size() << "\n";

    for (const auto& arg: args) {
        std::cout << "Arg: " << arg << "\n";
    }

    const std::string action = args.at(0);

    if (action == "improvise") {
        const std::string database_path = args.at(1);
        const std::string word = args.at(2);

        std::cout << "database_path: " << database_path << "\n";

        lc::VectorDatabase database {};

        database.load_file(database_path);

        std::cout << "Database loaded from path: " << database_path << "\n";
        std::cout << "Constructing text completer\n";

        lc::TextCompleter completer {std::move(database), 500, 500};

        std::cout << "Text completer created\n";

        const std::size_t nn_input_size = completer.word_vector_improviser_fields_sizes.total();
        const std::size_t nn_output_size = completer.word_vector_improviser_output_sizes.total();

        std::cout << "nn_input_size: " << nn_input_size << "\n";
        std::cout << "nn_output_size: " << nn_output_size << "\n";

        completer.set_word_vector_improviser_nn(
            {nn_input_size, 100, 100, 100, 100, 100, 100, 100, 100, 100, nn_output_size}, true);
        std::cout << "completer.word_vector_improviser.layer_sizes.back(): "
                  << completer.word_vector_improviser.layer_sizes.back() << "\n";

        std::cout << "Word vector improvser NN set\n";

        std::cout << "Word: " << word << "\n";

        const auto [res, type] = completer.find_word_vector(word);

        std::cout << "Word was improvised: " << res.improvised << "\n";
        std::cout << "Word vector quantity: " << lc::fancy_eigen_vector_str(res.word_vector.vector)
                  << "\n";
    };

    if (action == "context") {
        const std::string database_path = args.at(1);

        std::cout << "database_path: " << database_path << "\n";

        lc::VectorDatabase database {};

        database.load_file(database_path);

        std::cout << "Database loaded from path: " << database_path << "\n";

        lc::TextCompleter completer {std::move(database), 500, 500};

        std::cout << "Text completer created\n";

        const std::size_t nn_input_size = completer.context_builder_fields_sizes.total();
        const std::size_t nn_output_size = completer.context_builder_output_sizes.total();

        std::cout << "nn_input_size: " << nn_input_size << "\n";
        std::cout << "nn_output_size: " << nn_output_size << "\n";

        completer.set_context_builder_nn({nn_input_size, 1000, nn_output_size}, true);

        std::cout << "completer.context_builder.weights[0]: "
                  << lc::fancy_eigen_matrix_str(completer.context_builder.weights [0]) << "\n";

        std::cout << "completer.context_builder.biases[0]: "
                  << lc::fancy_eigen_vector_str(completer.context_builder.biases [0]) << "\n";

        // completer.ephemeral_memory = Eigen::VectorXf::Random(completer.ephemeral_memory.size());
        // completer.context_memory = Eigen::VectorXf::Random(completer.context_memory.size());

        std::cout << "Context builder NN set\n";

        float sentence_length_mean = 10.0F;
        float sentence_length_stddev = 5.0F;
        float flesch_kincaid_grade = 10.0F;

        completer.accumulate_context_memory(sentence_length_mean, sentence_length_stddev,
                                            flesch_kincaid_grade);

        std::cout << "Context memory accumulated\n";

        std::cout << "Context memory quantity: "
                  << lc::fancy_eigen_vector_str(completer.context_memory) << "\n";
    }
}
