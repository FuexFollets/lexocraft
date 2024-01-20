#include <ios>
#include <iostream>
#include <memory>

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

    const std::string database_path = args.at(0);
    const std::string word = args.at(1);

    std::cout << "database_path: " << database_path << "\n";

    lc::VectorDatabase database {};

    database.load(database_path);

    std::cout << "Database loaded from path: " << database_path << "\n";
    std::cout << "Constructing text completer\n";

    lc::TextCompleter completer {std::move(database), 500, 500};

    std::cout << "Text completer created\n";

    const std::size_t nn_input_size = completer.word_vector_improviser_fields_sizes.total();
    const std::size_t nn_output_size = completer.word_vector_improviser_output_sizes.total();

    std::cout << "nn_input_size: " << nn_input_size << "\n";
    std::cout << "nn_output_size: " << nn_output_size << "\n";

    completer.set_word_vector_improviser_nn({nn_input_size, 100, 100, 100, 100, 100, 100, 100, 100, 100, nn_output_size}, true);
    std::cout << "completer.word_vector_improviser.layer_sizes.back(): "
              << completer.word_vector_improviser.layer_sizes.back() << "\n";

    std::cout << "Word vector improvser NN set\n";

    std::cout << "Word: " << word << "\n";

    const auto res = completer.find_word_vector(word);

    std::cout << "Word was improvised: " << res.improvised << "\n";
    std::cout << "Word vector quantity: " << lc::fancy_eigen_vector_str(res.word_vector.vector)
              << "\n";
}
