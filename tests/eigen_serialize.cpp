#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry> // Needed for certain matrix types like Eigen::Quaternionf

// For serialization:
#include <Eigen/Eigen> // Includes the Serializer class
#include <cstdint>
#include <iostream>
#include <iterator>

int main() {
    using matrix_t = Eigen::Matrix3f;

    matrix_t matrix {matrix_t::Random()};

    std::cout << "Before serialization:\n" << matrix << "\n\n";

    Eigen::Serializer<matrix_t> serializer;

    auto size = serializer.size(matrix);

    auto* buffer = new std::uint8_t[size];

    serializer.serialize(buffer, std::next(buffer, size), matrix);

    std::cout << "After serialization:\n";
    for (std::size_t index {0}; index < size; ++index) {
        std::cout << static_cast<std::uint8_t>(buffer[index]) << " ";
    }

    matrix_t deserialized_matrix;
    serializer.deserialize(buffer, std::next(buffer, size), deserialized_matrix);

    std::cout << "\n\nAfter deserialization:\n" << deserialized_matrix << "\n";
}
