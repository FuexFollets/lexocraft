#ifndef LEXOCRAFT_CEREAL_ANNOY_HPP
#define LEXOCRAFT_CEREAL_ANNOY_HPP

#include <annoy/annoylib.h>
#include <cereal/archives/binary.hpp>
#include <cereal/cereal.hpp>
#include <cstdint>

namespace cereal {
    template <class Archive, typename S, typename T, typename Distance, typename Random,
              class ThreadedBuildPolicy>
    inline void
        save(Archive& archive,
             const Annoy::AnnoyIndex<S, T, Distance, Random, ThreadedBuildPolicy> annoy_index) {
        archive(annoy_index.serialize());
    }

    template <class Archive, typename S, typename T, typename Distance, typename Random,
              class ThreadedBuildPolicy>
    inline void load(Archive& archive,
                     Annoy::AnnoyIndex<S, T, Distance, Random, ThreadedBuildPolicy>& annoy_index) {
        std::string serialized_index;
        archive(serialized_index);

        const std::vector<uint8_t> bytes(serialized_index.begin(),
                                                           serialized_index.end());

        annoy_index.deserialize(bytes.data());
    }
} // namespace cereal

#endif
