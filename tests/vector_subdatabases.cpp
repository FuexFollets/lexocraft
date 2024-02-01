#include <iostream>
#include <memory>

#include <lexocraft/fancy_eigen_print.hpp>
#include <lexocraft/llm/text_completion.hpp>
#include <lexocraft/llm/vector_database.hpp>

int main(int argc, char** argv) {
    std::vector<std::string> args {std::next(argv, 1), std::next(argv, argc)};

    std::cout << "args: " << args.size() << "\n";

    for (const auto& arg: args) {
        std::cout << "Arg: " << arg << "\n";
    }

    const std::string action = args.at(0);

    if (action == "store") {
        const std::string vector_database_path = args.at(1);
        const std::string stored_path = args.at(2);

        lc::VectorDatabase database;

        std::cout << "Loading database from path: " << vector_database_path << '\n';
        database.load_file(vector_database_path);
        std::cout << "Successfully loaded vector database\n";

        lc::TextCompleter completer {std::move(database), 500, 500};

        std::cout << "Creating subvector databases\n";
        completer.create_subvector_databases();
        std::cout << "Subvector databases created\n";

        std::cout << "First word: " << completer.symbol_vector_subdatabase.words.at(0).word << "\n";

        std::cout << "Saving to path: " << stored_path << "\n";
        completer.save(stored_path);
        std::cout << "Subvector databases saved\n";
    }

    if (action == "load") {
        const std::string stored_path = args.at(1);
        lc::TextCompleter completer {lc::VectorDatabase {}, 500, 500};

        std::cout << "Loading from path: " << stored_path << "\n";

        completer.load(stored_path);

        std::cout << "Subvector databases loaded\n";

        /*
        for (const auto& subdatabase:
             {completer.alphanumeric_vector_subdatabase, completer.digit_vector_subdatabase,
              completer.symbol_vector_subdatabase, completer.acronym_vector_subdatabase,
              completer.homogeneous_vector_subdatabase}) {

            std::size_t index {};

            std::cout << "Subdatabase: " << subdatabase.words.size() << "\n";

            for (const auto& word: subdatabase.words) {
                std::cout << index++ << ": " << word.word << "\n";

                if (index > 10) {
                    break;
                }
            }
        }
        */

        for (const auto& word: completer.homogeneous_vector_subdatabase.words) {
            std::cout << word.word << "\n";
        }
    }
}
