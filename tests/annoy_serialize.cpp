#include <iostream>

#include <annoy/annoylib.h>
#include <annoy/kissrandom.h>

#include <lexocraft/cereal_annoy_index.hpp>

#include <random>

constexpr int DIMENSIONS = 10;

float random(float min, float max) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(min, max);
    return dis(gen);
}

void print_vector(float* vector) {
    for (int index {}; index < DIMENSIONS; ++index) {
        std::cout << vector [index] << " ";
    }
    std::cout << '\n';
}

int main() {
    // using AnnoyIndex_t = Annoy::AnnoyIndex<typename S, typename T, typename Distance, typename
    // Random, class ThreadedBuildPolicy>;

    using AnnoyIndex_t = Annoy::AnnoyIndex<int, float, Annoy::Euclidean, Annoy::Kiss64Random,
                                           Annoy::AnnoyIndexSingleThreadedBuildPolicy>;

    AnnoyIndex_t annoy_index(DIMENSIONS);

    for (int index {}; index < 1000; ++index) {
        float* vector = new float [DIMENSIONS];

        annoy_index.add_item(index, vector);
    }

    annoy_index.build(10);

    auto bytes = annoy_index.serialize();

    AnnoyIndex_t annoy_index2(DIMENSIONS);

    annoy_index2.deserialize(&bytes);

    std::cout << "annoy_index.get_n_items() = " << annoy_index.get_n_items() << '\n';
    std::cout << "annoy_index.get_n_trees() = " << annoy_index.get_n_trees() << '\n';

    std::cout << "annoy_index2.get_n_items() = " << annoy_index2.get_n_items() << '\n';
    std::cout << "annoy_index2.get_n_trees() = " << annoy_index2.get_n_trees() << '\n';

    std::cout << "(annoy_index.serialize() == annoy_index2.serialize()) = "
              << (annoy_index.serialize() == annoy_index2.serialize()) << '\n';
}
