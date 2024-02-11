#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include <cereal/archives/binary.hpp>
#include <cereal/cereal.hpp>
#include <icecream.hpp>
#include <nanobench.h>

#include <lexocraft/fancy_eigen_print.hpp>
#include <lexocraft/llm/text_completion.hpp>
#include <lexocraft/llm/vector_database.hpp>

std::string remove_whitespace(std::string str) {
    str.erase(str.begin(), std::find_if_not(str.begin(), str.end(), ::isspace));
    str.erase(std::find_if_not(str.rbegin(), str.rend(), ::isspace).base(), str.end());

    return str;
}

int main(const int argc, const char** argv) {
    std::vector<std::string> args {std::next(argv, 1), std::next(argv, argc)};

    std::cout << "args: " << args.size() << "\n";

    for (std::size_t index = 0; index < args.size(); ++index) {
        std::cout << "arg[" << index << "]: " << args [index] << "\n";
    }

    const std::string vector_database_type = args.at(0); // plaintext or binary
    const std::string vector_database_path = args.at(1);
    const std::string output_path = args.at(2);

    IC(vector_database_type);
    IC(vector_database_path);
    IC(output_path);

    std::optional<lc::VectorDatabase> vector_database {};

    if (vector_database_type == "plaintext") {
        std::cout << "Creating a plaintext vector database\n";
        std::ifstream vector_database_file {vector_database_path};
        std::vector<lc::WordVector> word_vectors {};
        std::string line {};

        IC();
        std::cout << "Reading word vectors from plaintext file: " << vector_database_path << "\n";
        while (std::getline(vector_database_file, line)) {
            word_vectors.emplace_back(remove_whitespace(line));
        }

        std::cout << "Read " << word_vectors.size() << " word vectors\n";

        vector_database.emplace(std::move(word_vectors));

        std::cout << "Building Annoy index\n";
        const auto result =
            ankerl::nanobench::Bench()
                .timeUnit(std::chrono::milliseconds {1}, "milliseconds")
                .epochs(1)
                .epochIterations(1)
                .run("build_annoy_index(t=5)", [&] { vector_database->build_annoy_index(5); });

        const auto elapsed_measure = ankerl::nanobench::Result::Measure::elapsed;
        std::cout << "Annoy index built in " << result.results().at(0).maximum(elapsed_measure)
                  << " seconds\n";
    }

    if (vector_database_type == "binary") {
        std::cout << "Creating a binary vector database\n";
        vector_database.emplace();

        std::cout << "Loading word vectors from binary file: " << vector_database_path << "\n";
        vector_database->load_file(vector_database_path);
        std::cout << "Loaded " << vector_database->words.size() << " word vectors\n";
    }

    std::cout << "Creating text completer\n";

    assert(vector_database.has_value());

    lc::TextCompleter text_completer {std::move(vector_database.value())};

    std::cout << "Creating vector subdatabases\n";
    text_completer.create_vector_subdatabases();

    std::cout << "Creating neural networks\n";
    text_completer.set_word_vector_improviser_layer_sizes(20);
    text_completer.set_context_builder_layer_sizes(20);
    text_completer.set_ephemeral_memory_accumulator_layer_sizes(20);

    std::vector<std::pair<std::string, std::shared_ptr<lc::VectorDatabase>>>
        vector_subdatabase_pairs {
            {"alphanumeric",           text_completer.alphanumeric_vector_subdatabase          },
            {"digit",                  text_completer.digit_vector_subdatabase                 },
            {"homogeneous",            text_completer.homogeneous_vector_subdatabase           },
            {"symbol",                 text_completer.symbol_vector_subdatabase                },
            {"lowercase_alphanumeric", text_completer.lowercase_alphanumeric_vector_subdatabase},
            {"lowercase_homogeneous",  text_completer.lowercase_homogeneous_vector_subdatabase },
    };

    for (const auto& [name, vector_subdatabase]: vector_subdatabase_pairs) {
        if (!vector_subdatabase->annoy_index_is_built) {
            const auto result =
                ankerl::nanobench::Bench()
                    .timeUnit(std::chrono::milliseconds {1}, "milliseconds")
                    .epochs(1)
                    .epochIterations(1)
                    .run(name + "(t=5)", [&] { vector_subdatabase->build_annoy_index(5); });
        }
    }

    std::cout << "Saving text completer to file: " << output_path << "\n";

    IC();
    IC(text_completer.word_vector_improviser_fields_sizes.word_vector_search_result);

    const auto result = text_completer.predict_next_token_value(
        lc::grammar::Token("sam", lc::grammar::Token::Type::Alphanumeric, false), 10, 10, 10, 10);

    IC(lc::fancy_eigen_vector_str(result.word_vector_value));
    IC(result.is_end);

    // ---------------------------- Test Serialization ----------------------------

    std::cout << "Testing serialization\n";

    std::stringstream text_completer_stream;

    cereal::BinaryOutputArchive output_archive {text_completer_stream};

    IC();
    std::cout << "Serializing text completer\n";
    output_archive(text_completer);

    std::cout << "Getting bytes from text completer\n";
    const std::string text_completer_serialized = text_completer_stream.str();
    std::cout << "Text completer serialized\n";

    cereal::BinaryInputArchive input_archive {text_completer_stream};

    IC();
    std::cout << "Deserializing text completer\n";
    input_archive(text_completer);
    std::cout << "Text completer deserialized\n";

    std::stringstream text_completer_stream2 {text_completer_serialized};
    cereal::BinaryOutputArchive output_archive2 {text_completer_stream2};

    std::cout << "Serializing deserialized text completer\n";
    output_archive2(text_completer);

    std::cout << "Getting bytes from deserialized text completer\n";
    const std::string deserialized_text_completer_serialized = text_completer_stream2.str();

    IC(text_completer_serialized.size());
    IC(deserialized_text_completer_serialized.size());
    IC(text_completer_serialized == deserialized_text_completer_serialized);

    std::cout << "Saving deserialized text completer to file: " << output_path << "\n";
    text_completer.save_file(output_path);
}
