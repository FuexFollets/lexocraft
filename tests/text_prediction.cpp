#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <icecream.hpp>

#include <lexocraft/fancy_eigen_print.hpp>
#include <lexocraft/llm/lexer.hpp>
#include <lexocraft/llm/text_completion.hpp>
#include <lexocraft/llm/vector_database.hpp>

int main(int argc, char** argv) {
    std::vector<std::string> args {std::next(argv, 1), std::next(argv, argc)};

    std::cout << "args: " << args.size() << "\n";

    for (const auto& arg: args) {
        std::cout << "Arg: " << arg << "\n";
    }

    const std::string vector_database_path = args.at(0);
    const std::string text_path = args.at(1);
    const std::string mode = args.at(2);

    // ------------------------- VectorDatabase -------------------------

    std::cout << "vector_database_path: " << vector_database_path << "\n";

    lc::VectorDatabase vector_database {};
    vector_database.load_file(vector_database_path);

    // ------------------------- TextCompleter -------------------------

    lc::TextCompleter completer {std::move(vector_database), 500, 500};

    const std::size_t improviser_input_size = completer.word_vector_improviser_fields_sizes.total();
    const std::size_t improviser_output_size =
        completer.word_vector_improviser_output_sizes.total();

    completer.set_word_vector_improviser_nn(
        {improviser_input_size, 200, 200, improviser_output_size}, true);

    const std::size_t predictor_input_size = completer.ephemeral_memory_fields_sizes.total();
    const std::size_t predictor_output_size = completer.ephemeral_memory_output_sizes.total();

    completer.set_ephemeral_memory_accmulator_nn(
        {predictor_input_size, 200, 200, predictor_output_size}, true);

    completer.create_vector_subdatabases();

    IC();
    IC(completer.homogeneous_vector_subdatabase.word_map.size());

    // ------------------------- Text -------------------------

    std::cout << "Loading text from path: " << text_path << "\n";
    const std::ifstream file {text_path};
    std::stringstream file_contents_buffer;
    file_contents_buffer << file.rdbuf();
    const std::string file_contents {file_contents_buffer.str()};
    std::cout << "Text loaded from path: " << text_path << "\n";

    std::cout << "\n\nText: " << file_contents << "\n";

    const std::vector<lc::grammar::Token> tokens =
        lc::grammar::tokenize(file_contents, completer.vector_database);

    const float sentence_length_mean_ = 10.0F;
    const float sentence_length_stddev_ = 7.0F;
    const float flesch_kincaid_grade_ = 5.0F;
    const float sentence_count_ = 5.0F;

    // ------------------------- Predict -------------------------

    for (const lc::grammar::Token& token: tokens) {
        completer.set_word_vector_improviser_nn(
            {improviser_input_size, 200, 200, improviser_output_size}, true);

        completer.set_ephemeral_memory_accmulator_nn(
            {predictor_input_size, 200, 200, predictor_output_size}, true);

        std::cout << "\nToken: " << token << "\n";

        lc::TextCompleter::EphemeralMemoryNNOutput output {completer.predict_next_token_value(
            token, sentence_length_mean_, sentence_length_stddev_, flesch_kincaid_grade_,
            sentence_count_)};

        std::cout << "Ephemeral memory: " << lc::fancy_eigen_vector_str(output.ephemeral_memory)
                  << "\n";
        std::cout << "Predicted word vector value: "
                  << lc::fancy_eigen_vector_str(output.word_vector_value) << "\n";

        const std::vector<lc::VectorDatabase::SearchResult> results =
            completer.vector_database.search_closest_vector_value_n(output.word_vector_value, 10);

        std::cout << "Closest " << results.size() << " results\n";

        for (const auto& result: results) {
            std::cout << std::setprecision(2) << "Similarity: " << result.similarity << " "
                      << result.word.word << " " << lc::fancy_eigen_vector_str(result.word.vector)
                      << "\n";
        }

        if (mode == "manual") {
            std::string input;

            std::getline(std::cin, input);
        }

        std::cout << "-------------------\n\n";
    }
}
