#include "cereal/archives/binary.hpp"
#include <iostream>

#include <annoy/annoylib.h>
#include <annoy/kissrandom.h>
#include <cereal/cereal.hpp>

#include <random>
#include <sstream>

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

    std::stringstream sstream;

    cereal::BinaryOutputArchive oarchive {sstream};

    oarchive(annoy_index);

    cereal::BinaryInputArchive iarchive {sstream};

    AnnoyIndex_t loaded_annoy_index;

    iarchive(loaded_annoy_index);

    std::cout << "Loaded Annoy Index\n";

    const std::size_t num_items = annoy_index.get_n_items();
    const std::size_t num_items_loaded = loaded_annoy_index.get_n_items();

    std::cout << "Number of items in original index: " << num_items << '\n';
    std::cout << "Number of items in loaded index: " << num_items_loaded << '\n';

    std::cout << "Original Annoy Index\n";
    for (int index {}; index < 10; ++index) {
        float* vector = new float [DIMENSIONS];
        annoy_index.get_item(index, vector);
        print_vector(vector);

        delete [] vector;
    }

    std::cout << "\n\nLoaded Annoy Index\n";
    for (int index {}; index < 10; ++index) {
        float* vector = new float [DIMENSIONS];
        loaded_annoy_index.get_item(index, vector);
        print_vector(vector);

        delete [] vector;
    }
}
