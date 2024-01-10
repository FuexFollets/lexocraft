#ifndef TEXT_COMPLETION_HPP
#define TEXT_COMPLETION_HPP

#include <Eigen/Eigen>

#include <lexocraft/llm/vector_database.hpp>
#include <lexocraft/neural_network/neural_network.hpp>

namespace lc {
    class TextCompleter {
        public:

        Eigen::VectorXf ephemeral_memory;
        Eigen::VectorXf context_memory;

        NeuralNetwork ephemeral_memory_accmulator;
        NeuralNetwork context_builder;
        VectorDatabase* vector_database;

        struct EphemeralMemoryNNFields {
            /* TODO: Vector fields for EphemeralMemoryNN */
        };

        struct EphemeralMemoryNNOutput {
            /* TODO: Vector fields for EphemeralMemoryNN output */
        };

        struct ContextBuilderNNFields {
            /* TODO: Vector fields for ContextBuilderNN */
        };

        struct ContextBuilderNNOutput {
            /* TODO: Vector fields for ContextBuilderNN output */
        };
    };

} // namespace lc

#endif // TEXT_COMPLETION_HPP
