#include <string>
#include <vector>

#include <lexocraft/llm/vector_database.hpp>

int main(int argc, char** argv) {
    std::vector<std::string> args {std::next(argv, 1), std::next(argv, argc)};

    std::cout << "args: " << args.size() << "\n";
    for (const auto& arg: args) {
        std::cout << arg << "\n";
    }

    if (args.at(0) == "create") {
        const std::string path_from = args.at(1);
        const std::string path_to = args.at(2);

        std::ifstream file {path_from};
        std::vector<lc::WordVector> words;
        std::string line;

        std::cout << "Reading from file: " << path_from << "\n";

        while (std::getline(file, line)) {
            words.emplace_back(std::move(line));
        }

        std::cout << "Read " << words.size() << " words from file\n";

        lc::VectorDatabase database {words};

        database.save(path_to);

        std::cout << "Database saved to path: " << path_to << "\n";

        std::cout << "First 20 words:\n";

        for (std::size_t index {0}; index < 20 && index < database.words.size(); ++index) {
            std::cout << database.words [index].word << "\n";
        }
    }

    if (args.at(0) == "load") {
        const std::string path = args.at(1);

        lc::VectorDatabase database;

        std::cout << "Loading database from path: " << path << "\n";

        database.load(path);

        std::cout << "Database loaded from path: " << path << "\n";

        std::cout << "First 20 words:\n";

        for (std::size_t index {0}; index < 20 && index < database.words.size(); ++index) {
            std::cout << database.words [index].word << ;
        }
    }
}
